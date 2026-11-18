#ifndef INFINI_OPS_KUNLUN_LAUNCH2_DECLARATIONS_H_
#define INFINI_OPS_KUNLUN_LAUNCH2_DECLARATIONS_H_

// Host-side declarations for the batch-2 Kunlun kernels implemented in
// `kernels.cu` (compiled by the XTDK). Included by the per-op wrappers.

namespace infini::ops {
namespace kunlun_launch {

// tag: 0=f32, 1=f16, 2=bf16 (see elementwise_common.h KunlunDtypeTag).
void LaunchSigmoid(int tag, void* out, const void* in, unsigned long long total,
                   void* stream);
void LaunchSigmoidStrided(
    int tag, void* out, const void* in, const unsigned long long* out_shape,
    const unsigned long long* in_shape, const long long* out_strides,
    const long long* in_strides, unsigned long long output_size,
    unsigned long long ndim, bool out_c, bool in_c, void* stream);
void LaunchRmsNorm(int tag, void* out, const void* in, const void* w,
                   unsigned int rows, int dim, float eps,
                   long long in_row_stride, void* stream);
void LaunchFusedAddRmsNorm(int tag, void* in_out, void* residual, const void* w,
                           unsigned int rows, int dim, float eps,
                           long long in_row_stride,
                           long long residual_row_stride, void* stream);
void LaunchSoftmax(int tag, void* out, const void* in, unsigned int rows,
                   int dim, long long in_row_stride, long long out_row_stride,
                   void* stream);
void LaunchEmbedding(int tag, int index_is_64, void* out, const void* indices,
                     const void* w, unsigned long long total, int emb_dim,
                     unsigned long long vocab, void* stream);
void LaunchCopy(int element_size, void* out, const void* in, int ndim,
                const long long* shape, const long long* in_strides,
                const long long* out_strides, unsigned long long total,
                void* stream);
void LaunchNarrowI64ToI32(void* out, const void* in, unsigned long long total,
                          void* stream);
void LaunchTransposeCacheBlocks(int tag, void* out, const void* in,
                                int num_blocks, int page_size, int kv_heads,
                                int dim, void* stream);
void LaunchTransposeBroadcast(int tag, void* out, const void* in,
                              int num_blocks, int page_size, int kv_heads,
                              int q_heads, int dim, void* stream);
void LaunchGatherKv(int tag, void* out, const void* cache, const void* kvlod,
                    const void* block_tables, int kv_heads, int dim,
                    int page_size, int bt_stride, int batch, int kv_tokens,
                    void* stream);
void LaunchGatherKv(int tag, void* out, const void* cache, const void* kvlod,
                    const void* block_tables, int kv_heads, int dim,
                    int page_size, int bt_stride, int batch, int kv_tokens,
                    void* stream);
void LaunchCast(int tag_in, int tag_out, void* out, const void* in,
                unsigned long long total, void* stream);
void LaunchFill(int element_size, int tag, void* out, int ndim,
                const long long* shape, const long long* out_strides,
                unsigned long long total, float value, void* stream);
void LaunchTriangular(bool upper, int element_size, void* out, const void* in,
                      int ndim, const long long* shape,
                      const long long* in_strides, const long long* out_strides,
                      int rows, unsigned long long total, int diagonal,
                      void* stream);
void LaunchArgmax(int tag, void* out, const void* in, unsigned int rows,
                  int dim, long long row_stride, void* stream);
void LaunchRope(int tag, bool interleaved, void* out, const void* in,
                const void* positions, const void* sin_table,
                const void* cos_table, int half, long long x_stride_seq,
                long long x_stride_head, long long y_stride_seq,
                long long y_stride_head, unsigned long long seq,
                unsigned long long heads, unsigned long long batch,
                long long x_stride_batch, long long y_stride_batch,
                bool batched_pos, void* stream);

void LaunchScaleF32(void* y, unsigned long long total, float alpha,
                    void* stream);

void LaunchKlGemmTile(void* y, const void* a, const void* b, int m, int n,
                      int k, int lda, int ldb, void* stream);
void LaunchGatherBroadcastKv(void* out, const void* cache,
                             const void* block_table, int bt_stride,
                             int page_size, int kv_heads, int q_heads, int dim,
                             int clen, void* stream);
void LaunchGatherDenseBroadcastKV(void* k_out, void* v_out, const void* k_cache,
                                  const void* v_cache, const void* block_table,
                                  int bt_stride, int page_size, int kv_heads,
                                  int q_heads, int dim, int clen, void* stream);
void LaunchGatherDenseBroadcast(void* out, const void* cache,
                                const void* block_table, int bt_stride,
                                int page_size, int kv_heads, int q_heads,
                                int dim, int clen, void* stream);
void LaunchGqaBroadcast(void* out, const void* in, int tokens, int kv_heads,
                        int q_heads, int dim, void* stream);
void LaunchGatherPage(void* out, const void* cache, const void* block_table,
                      int bt_stride, int page_size, int kv_heads, int dim,
                      int clen, int tag, void* stream);
void LaunchBroadcastGqa(void* out, const void* in, int tokens, int kv_heads,
                        int q_heads, int dim, int tag, void* stream);
void LaunchQK(void* scores, const void* q, const void* k, float scale, int clen,
              int heads, int dim, int tag, void* stream);
void LaunchPv(void* out, const void* scores, const void* v, int clen, int heads,
              int dim, int tag, void* stream);
}  // namespace kunlun_launch
}  // namespace infini::ops

#endif
