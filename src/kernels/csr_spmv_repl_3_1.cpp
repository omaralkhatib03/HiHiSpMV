#include "xlx_definitions.hpp"

/*
    (Multi-tile) CSR SpMV Logic split into four kernels.
        - Reader/Writer
        - Multiplier
        - Partial summer 
        - Accumulator
*/

void mark_row_elements(
        hls::stream<indmsk_t> &out_row_marks,
        hls::stream<pkt_ind_nnz> &in_row_info,
        const unsigned int tiles,
        const unsigned int runs) {

#if DEBUG
    if (DEBUG&1) printf ("k3::mark_row_elements(): start\n");
#endif

    assert(runs>0);
    runs:
    for (unsigned int h=0; h<runs; h++) {
        #pragma HLS PIPELINE OFF

        assert(tiles>0);
        tiles:
        for (unsigned int tile=0; tile<tiles; tile++) {
            #pragma HLS PIPELINE OFF
            #pragma HLS LOOP_FLATTEN OFF
            #pragma HLS loop_tripcount min=(tiles_min) max=(tiles_max)

            indind_t row_item {.index=-1, .value=-1, .is_last=false};
            int block_ind=BLOCK_SIZE;

            row_marking:
            for (/*int i=0*/; true; /*i++*/) {
                #pragma HLS PIPELINE II=1

                bool row_read = row_item.value <= 0;
                pkt_ind_nnz v = row_read ? in_row_info.read() : pkt_ind_nnz();
                row_read ? row_item.get(v.data) : 0;

                if (row_item.is_last) break;

                bool block_read = block_ind > BLOCK_SIZE-1;
                block_ind *= !block_read;

                int start = block_ind;
                int bound = row_item.value-1 + block_ind;
                bool write = bound < BLOCK_SIZE;
                int end = bound < BLOCK_SIZE? bound : BLOCK_SIZE-1;
                ap_uint<BLOCK_SIZE> one_16 = 0xFFFF;
                ap_uint<BLOCK_SIZE> mask =  one_16 >> start & one_16 << (BLOCK_SIZE-1-end);

                indmsk_t row_mark {.index=row_item.index, .mask=mask, .is_write=write, .is_last=false, .is_blk_read=block_read};
                
                out_row_marks.write(row_mark);
                int old_nnz = row_item.value;
                row_item.value -= BLOCK_SIZE-block_ind;
                block_ind += old_nnz;
            }

            indmsk_t row_mark {.index=row_item.index, .is_write=false, .is_last=true, .is_blk_read=false};
            out_row_marks.write(row_mark);
        }
    }
}

void data_prefix_sum(
        hls::stream<pkt_ind_val> &out_rows,
        hls::stream<indmsk_t> &in_row_marks,
        hls::stream<pkt_block> &in_data, 
        const unsigned int tiles,
        const unsigned int nnz_blocks,
        const unsigned int runs) {
    
    #pragma HLS INLINE

    assert(runs>0);
    runs:
    for (unsigned int h=0; h<runs; h++) {
        #pragma HLS PIPELINE OFF
        
        assert(tiles>0);
        tiles:
        for (unsigned int tile=0; tile<tiles; tile++) {
            #pragma HLS PIPELINE OFF
            #pragma HLS LOOP_FLATTEN OFF
            #pragma HLS loop_tripcount min=(tiles_min) max=(tiles_max)

            valvec_k2k_t prod_block;
            #pragma HLS array_partition variable=prod_block.items complete dim=0

            int block_ind = BLOCK_SIZE;
            row_sum:
            for (/*int i=0*/; true; /*i++*/) {
                #pragma HLS PIPELINE II=1
                indmsk_t row_mark = in_row_marks.read();

                if (row_mark.is_last) break;

                pkt_block v2 = row_mark.is_blk_read ? in_data.read() : pkt_block();
                row_mark.is_blk_read ? prod_block.get(v2.data) : 0;

                valvec_k2k_t res_block;
                prec_t sum_block = 0;
                #pragma HLS BIND_OP variable=sum_block op=fadd impl=fulldsp
                for (unsigned int j=0; j<BLOCK_SIZE; j++) 
                {
                    #pragma HLS UNROLL
                    bool mark = row_mark.mask.test(BLOCK_SIZE-1-j);
                    sum_block += mark ? prod_block.items[j] : 0;
                }

                indval_t row_res {.index=row_mark.index, .value=sum_block, .is_last=false, .is_write=row_mark.is_write};
                pkt_ind_val v3;
                row_res.set(v3.data);
                out_rows.write(v3); 
            }

            indval_t row_res {.index=VECTOR_SIZE+BLOCK_SIZE-1, .value=0, .is_last=true, .is_write=false};
            pkt_ind_val v4;
            row_res.set(v4.data);
            out_rows.write(v4);       
 
        }
    }  
}   

    
extern "C" {

    void csr_spmv_repl_3(
            hls::stream<pkt_ind_val> &out_rows,
            hls::stream<pkt_ind_nnz> &in_row_tupples,
            hls::stream<pkt_block> &in_prod,
            const unsigned int tiles,
            const unsigned int nnz_blocks_tot,
            const unsigned int runs) {
        
        #pragma HLS INTERFACE axis port = out_rows
        #pragma HLS INTERFACE axis port = in_row_tupples
        #pragma HLS INTERFACE axis port = in_prod

        #pragma HLS INTERFACE s_axilite port = tiles
        #pragma HLS INTERFACE s_axilite port = nnz_blocks_tot
        #pragma HLS INTERFACE s_axilite port = runs

        const unsigned int str_depth = 16; //
        static hls::stream<indmsk_t> row_marks;
        #pragma HLS STREAM variable=row_marks depth=str_depth 

        #pragma HLS DATAFLOW

        mark_row_elements(row_marks, in_row_tupples, tiles, runs);
        data_prefix_sum(out_rows, row_marks, in_prod, tiles, nnz_blocks_tot, runs);      
    }
}