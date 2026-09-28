#include "myds_engram.h"
#include "ds4_engram.h"
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
 if(argc!=2)return 2;
 gguf_file_t *gf=gguf_open(argv[1]); assert(gf);
 const gguf_tensor_t *emb=gguf_find_tensor(gf,"token_embd.weight"); assert(emb);
 myds_engram_t *native=myds_engram_create(gf,argv[1],(uint32_t)emb->dims[1],(uint32_t)emb->dims[0]);
 if(!native){perror("native create");return 1;}
 size_t before=gf->mmap_size;
 if(!myds_engram_unmap_tables(gf,native)){perror("unmap");return 1;}
 printf("mapped_bytes_before=%zu mapped_bytes_after=%zu removed=%zu\n",before,gf->mmap_size,before-gf->mmap_size);
 ds4_engram_layout layout={.token_map=native->layout.token_map,.vocab_size=native->layout.vocab_size,.compressed_vocab_size=native->layout.compressed_vocab_size,.pad_id=native->layout.pad_id};
 memcpy(layout.rows,native->layout.rows,sizeof(layout.rows));
 memcpy(layout.primes,native->layout.primes,sizeof(layout.primes));
 memcpy(layout.multipliers,native->layout.multipliers,sizeof(layout.multipliers));
 assert(ds4_engram_layout_valid(&layout));
 ds4_engram_history history; ds4_engram_history_reset(&history);
 float values[2][24*256]; uint32_t ids[2][24];
 uint64_t hash=UINT64_C(1469598103934665603); size_t nonzero=0;
 for(int pos=0;pos<9;pos++) {
  int token=42+pos;
  assert(myds_engram_prepare_token(native,token,(uint64_t)pos));
  assert(ds4_engram_hash(&layout,&history,&token,NULL,1,&ids[0][0]));
  assert(!memcmp(ids,native->row_ids,sizeof(ids)));
  for(int layer=0;layer<2;layer++) {
   ds4_engram_table table={.fd=native->table[layer].fd,.offset=native->table[layer].offset,.rows=native->table[layer].rows};
   assert(ds4_engram_read(&table,ids[layer],24,values[layer]));
   assert(!memcmp(values[layer],native->row_values[layer],sizeof(values[layer])));
   for(int j=0;j<24*256;j++)nonzero+=values[layer][j]!=0;
  }
  const unsigned char *bytes=(const unsigned char*)values;
  for(size_t i=0;i<sizeof(values);i++){hash^=bytes[i];hash*=UINT64_C(1099511628211);}
 }
 printf("tokens=9 row_ids_compared=432 f32_values_bitwise_equal=110592 nonzero=%zu fnv1a64=%016" PRIx64 " row_io_upper_bound_bytes=228096\n",nonzero,hash);
 myds_engram_free(native);gguf_close(gf);return 0;
}
