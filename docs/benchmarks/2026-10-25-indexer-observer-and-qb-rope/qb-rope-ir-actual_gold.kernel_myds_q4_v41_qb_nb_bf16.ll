source_filename = "kernel_myds_q4_v41_qb_nb_bf16"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v27-apple-macosx15.0.0"

%struct.ds4_metal_args_mul_mv = type { i32, i32, i32, i64, i64, i64, i64, i32, i32, i32, i64, i64, i64, i64, i32, i32, i32, i16, i16 }

@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_myds_all.metal, i8* null }]
@_ZL13FC_mul_mv_nsg = internal unnamed_addr addrspace(2) global i16 undef, align 2
@_Z13FC_mul_mv_nsg.MTL_FC_INIT_600_s = internal unnamed_addr addrspace(2) externally_initialized constant i16 undef, section "air.fc_initializer", align 2

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn writeonly
define internal void @_GLOBAL__sub_I_myds_all.metal() #0 section "air.static_init" {
  %1 = load i16, i16 addrspace(2)* @_Z13FC_mul_mv_nsg.MTL_FC_INIT_600_s, align 2, !tbaa !28
  store i16 %1, i16 addrspace(2)* @_ZL13FC_mul_mv_nsg, align 2, !tbaa !28
  ret void
}

; Function Attrs: convergent mustprogress nounwind
define void @kernel_myds_q4_v41_qb_nb_bf16(%struct.ds4_metal_args_mul_mv addrspace(2)* nocapture noundef readnone align 8 dereferenceable(112) "air-buffer-no-alias" %0, i8 addrspace(1)* nocapture noundef readonly "air-buffer-no-alias" %1, i8 addrspace(1)* nocapture noundef readonly "air-buffer-no-alias" %2, i8 addrspace(1)* nocapture noundef writeonly "air-buffer-no-alias" %3, i8 addrspace(3)* nocapture noundef readnone "air-buffer-no-alias" %4, <3 x i32> noundef %5, i16 noundef %6, i16 noundef %7) local_unnamed_addr #1 {
  %9 = alloca [16 x float], align 4
  %10 = alloca [16 x float], align 4
  %11 = alloca i64, align 8
  %12 = bitcast i64* %11 to [2 x float]*
  %13 = load i16, i16 addrspace(2)* @_ZL13FC_mul_mv_nsg, align 2, !tbaa !28
  %14 = extractelement <3 x i32> %5, i64 0
  %15 = sext i16 %13 to i32
  %16 = mul i32 %14, %15
  %17 = zext i16 %7 to i32
  %18 = add i32 %16, %17
  %19 = shl i32 %18, 1
  %20 = mul i32 %18, 1440
  %21 = bitcast [16 x float]* %9 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %21) #5
  %22 = bitcast [16 x float]* %10 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %22) #5
  %23 = bitcast i64* %11 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %23) #5
  store i64 0, i64* %11, align 8
  %24 = icmp ult i16 %6, 40
  br i1 %24, label %25, label %51

25:                                               ; preds = %8
  %26 = lshr i16 %6, 3
  %27 = zext i16 %26 to i32
  %28 = bitcast i8 addrspace(1)* %2 to float addrspace(1)*
  %29 = shl nuw nsw i32 %27, 8
  %30 = zext i32 %29 to i64
  %31 = getelementptr inbounds float, float addrspace(1)* %28, i64 %30
  %32 = lshr i16 %6, 2
  %33 = and i16 %32, 1
  %34 = zext i16 %33 to i32
  %35 = shl nuw nsw i32 %34, 6
  %36 = zext i32 %35 to i64
  %37 = getelementptr inbounds float, float addrspace(1)* %31, i64 %36
  %38 = and i16 %6, 3
  %39 = zext i16 %38 to i32
  %40 = shl nuw nsw i32 %39, 3
  %41 = zext i32 %40 to i64
  %42 = getelementptr inbounds float, float addrspace(1)* %37, i64 %41
  %43 = zext i16 %33 to i64
  %44 = shl nuw nsw i32 %34, 4
  %45 = zext i32 %44 to i64
  %46 = shl nuw nsw i32 %39, 2
  %47 = zext i32 %46 to i64
  br label %48

48:                                               ; preds = %105, %25
  %49 = phi float addrspace(1)* [ %42, %25 ], [ %106, %105 ]
  %50 = phi i32 [ %27, %25 ], [ 4, %105 ]
  br label %67

51:                                               ; preds = %105, %8
  %52 = bitcast i8 addrspace(1)* %3 to float addrspace(1)*
  %53 = icmp eq i16 %6, 0
  br label %280

54:                                               ; preds = %67
  %55 = mul nuw nsw i32 %50, 144
  %56 = add i32 %55, %20
  %57 = zext i32 %56 to i64
  %58 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %57
  %59 = getelementptr inbounds i8, i8 addrspace(1)* %58, i64 4
  %60 = bitcast i8 addrspace(1)* %59 to i16 addrspace(1)*
  %61 = getelementptr inbounds i16, i16 addrspace(1)* %60, i64 %43
  %62 = getelementptr inbounds i8, i8 addrspace(1)* %58, i64 16
  %63 = bitcast i8 addrspace(1)* %62 to i16 addrspace(1)*
  %64 = getelementptr inbounds i16, i16 addrspace(1)* %63, i64 %45
  %65 = getelementptr inbounds i16, i16 addrspace(1)* %64, i64 %47
  %66 = bitcast i8 addrspace(1)* %58 to half addrspace(1)*
  br label %108

67:                                               ; preds = %67, %48
  %68 = phi i32 [ 0, %48 ], [ %103, %67 ]
  %69 = phi <4 x float> [ zeroinitializer, %48 ], [ %102, %67 ]
  %70 = zext i32 %68 to i64
  %71 = getelementptr inbounds float, float addrspace(1)* %49, i64 %70
  %72 = load float, float addrspace(1)* %71, align 4, !tbaa !32, !alias.scope !34, !noalias !37
  %73 = getelementptr inbounds [16 x float], [16 x float]* %9, i64 0, i64 %70
  store float %72, float* %73, align 4, !tbaa !32
  %74 = extractelement <4 x float> %69, i64 0
  %75 = fadd fast float %74, %72
  %76 = insertelement <4 x float> poison, float %75, i64 0
  %77 = add nuw nsw i32 %68, 32
  %78 = zext i32 %77 to i64
  %79 = getelementptr inbounds float, float addrspace(1)* %49, i64 %78
  %80 = load float, float addrspace(1)* %79, align 4, !tbaa !32, !alias.scope !34, !noalias !37
  %81 = add nuw nsw i32 %68, 8
  %82 = zext i32 %81 to i64
  %83 = getelementptr inbounds [16 x float], [16 x float]* %9, i64 0, i64 %82
  store float %80, float* %83, align 4, !tbaa !32
  %84 = extractelement <4 x float> %69, i64 1
  %85 = fadd fast float %80, %84
  %86 = insertelement <4 x float> %76, float %85, i64 1
  %87 = add nuw nsw i32 %68, 128
  %88 = zext i32 %87 to i64
  %89 = getelementptr inbounds float, float addrspace(1)* %49, i64 %88
  %90 = load float, float addrspace(1)* %89, align 4, !tbaa !32, !alias.scope !34, !noalias !37
  %91 = getelementptr inbounds [16 x float], [16 x float]* %10, i64 0, i64 %70
  store float %90, float* %91, align 4, !tbaa !32
  %92 = extractelement <4 x float> %69, i64 2
  %93 = fadd fast float %90, %92
  %94 = insertelement <4 x float> %86, float %93, i64 2
  %95 = add nuw nsw i32 %68, 160
  %96 = zext i32 %95 to i64
  %97 = getelementptr inbounds float, float addrspace(1)* %49, i64 %96
  %98 = load float, float addrspace(1)* %97, align 4, !tbaa !32, !alias.scope !34, !noalias !37
  %99 = getelementptr inbounds [16 x float], [16 x float]* %10, i64 0, i64 %82
  store float %98, float* %99, align 4, !tbaa !32
  %100 = extractelement <4 x float> %69, i64 3
  %101 = fadd fast float %98, %100
  %102 = insertelement <4 x float> %94, float %101, i64 3
  %103 = add nuw nsw i32 %68, 1
  %104 = icmp eq i32 %103, 8
  br i1 %104, label %54, label %67, !llvm.loop !40

105:                                              ; preds = %127
  %106 = getelementptr inbounds float, float addrspace(1)* %49, i64 1024
  %107 = icmp eq i32 %50, 0
  br i1 %107, label %48, label %51, !llvm.loop !42

108:                                              ; preds = %127, %54
  %109 = phi i16 [ 0, %54 ], [ %193, %127 ]
  %110 = phi half addrspace(1)* [ %66, %54 ], [ %192, %127 ]
  %111 = phi i16 addrspace(1)* [ %65, %54 ], [ %190, %127 ]
  %112 = phi i16 addrspace(1)* [ %61, %54 ], [ %191, %127 ]
  %113 = load i16, i16 addrspace(1)* %112, align 2, !tbaa !28, !alias.scope !43, !noalias !44
  %114 = lshr i16 %113, 8
  %115 = getelementptr inbounds i16, i16 addrspace(1)* %112, i64 2
  %116 = load i16, i16 addrspace(1)* %115, align 2, !tbaa !28, !alias.scope !43, !noalias !44
  %117 = lshr i16 %116, 8
  %118 = getelementptr inbounds i16, i16 addrspace(1)* %112, i64 4
  %119 = load i16, i16 addrspace(1)* %118, align 2, !tbaa !28, !alias.scope !43, !noalias !44
  %120 = and i16 %119, 3855
  %121 = lshr i16 %113, 2
  %122 = and i16 %121, 12336
  %123 = or i16 %120, %122
  %124 = lshr i16 %123, 8
  %125 = lshr i16 %116, 2
  %126 = getelementptr inbounds i16, i16 addrspace(1)* %111, i64 32
  br label %195

127:                                              ; preds = %195
  %128 = and i16 %114, 63
  %129 = and i16 %117, 63
  %130 = lshr i16 %119, 4
  %131 = and i16 %125, 12336
  %132 = zext i16 %128 to i32
  %133 = zext i16 %129 to i32
  %134 = zext i16 %124 to i32
  %135 = and i16 %130, 3855
  %136 = or i16 %135, %131
  %137 = load half, half addrspace(1)* %110, align 2, !tbaa !45, !alias.scope !43, !noalias !44
  %138 = fpext half %137 to float
  %139 = fmul fast float %221, 3.906250e-03
  %140 = fadd fast float %139, %211
  %141 = and i16 %113, 63
  %142 = zext i16 %141 to i32
  %143 = tail call fast float @air.convert.f.f32.s.i32(i32 %142) #6
  %144 = fmul fast float %143, %140
  %145 = fmul fast float %241, 3.906250e-03
  %146 = fadd fast float %145, %231
  %147 = tail call fast float @air.convert.f.f32.s.i32(i32 %132) #6
  %148 = fmul fast float %147, %146
  %149 = fmul fast float %260, 3.906250e-03
  %150 = fadd fast float %149, %252
  %151 = and i16 %123, 63
  %152 = zext i16 %151 to i32
  %153 = tail call fast float @air.convert.f.f32.s.i32(i32 %152) #6
  %154 = fmul fast float %153, %150
  %155 = fmul fast float %276, 3.906250e-03
  %156 = fadd fast float %155, %268
  %157 = tail call fast float @air.convert.f.f32.s.i32(i32 %134) #6
  %158 = fmul fast float %157, %156
  %159 = fadd fast float %158, %148
  %160 = fmul fast float %159, 6.250000e-02
  %161 = fadd fast float %154, %144
  %162 = fadd fast float %161, %160
  %163 = fmul fast float %162, %138
  %164 = getelementptr inbounds half, half addrspace(1)* %110, i64 1
  %165 = load half, half addrspace(1)* %164, align 2, !tbaa !45, !alias.scope !43, !noalias !44
  %166 = fpext half %165 to float
  %167 = and i16 %116, 63
  %168 = zext i16 %167 to i32
  %169 = tail call fast float @air.convert.f.f32.s.i32(i32 %168) #6
  %170 = fmul fast float %169, %75
  %171 = tail call fast float @air.convert.f.f32.s.i32(i32 %133) #6
  %172 = fmul fast float %171, %85
  %173 = fadd fast float %172, %170
  %174 = and i16 %136, 63
  %175 = zext i16 %174 to i32
  %176 = tail call fast float @air.convert.f.f32.s.i32(i32 %175) #6
  %177 = fmul fast float %176, %93
  %178 = fadd fast float %173, %177
  %179 = lshr i16 %136, 8
  %180 = zext i16 %179 to i32
  %181 = tail call fast float @air.convert.f.f32.s.i32(i32 %180) #6
  %182 = fmul fast float %181, %101
  %183 = fadd fast float %178, %182
  %184 = zext i16 %109 to i64
  %185 = getelementptr inbounds [2 x float], [2 x float]* %12, i64 0, i64 %184
  %186 = load float, float* %185, align 4, !tbaa !32
  %187 = fadd fast float %186, %163
  %188 = fmul fast float %183, %166
  %189 = fsub fast float %187, %188
  store float %189, float* %185, align 4, !tbaa !32
  %190 = getelementptr inbounds i16, i16 addrspace(1)* %111, i64 360
  %191 = getelementptr inbounds i16, i16 addrspace(1)* %112, i64 360
  %192 = getelementptr inbounds half, half addrspace(1)* %110, i64 360
  %193 = add nuw nsw i16 %109, 1
  %194 = icmp eq i16 %193, 2
  br i1 %194, label %105, label %108, !llvm.loop !47

195:                                              ; preds = %195, %108
  %196 = phi i32 [ 0, %108 ], [ %278, %195 ]
  %197 = phi <4 x float> [ zeroinitializer, %108 ], [ %277, %195 ]
  %198 = phi <4 x float> [ zeroinitializer, %108 ], [ %242, %195 ]
  %199 = shl nuw i32 %196, 1
  %200 = zext i32 %199 to i64
  %201 = getelementptr inbounds [16 x float], [16 x float]* %9, i64 0, i64 %200
  %202 = load float, float* %201, align 4, !tbaa !32
  %203 = zext i32 %196 to i64
  %204 = getelementptr inbounds i16, i16 addrspace(1)* %111, i64 %203
  %205 = load i16, i16 addrspace(1)* %204, align 2, !tbaa !28, !alias.scope !43, !noalias !44
  %206 = zext i16 %205 to i32
  %207 = and i32 %206, 15
  %208 = tail call fast float @air.convert.f.f32.s.i32(i32 %207) #6
  %209 = fmul fast float %208, %202
  %210 = extractelement <4 x float> %198, i64 0
  %211 = fadd fast float %209, %210
  %212 = insertelement <4 x float> poison, float %211, i64 0
  %213 = or i32 %199, 1
  %214 = zext i32 %213 to i64
  %215 = getelementptr inbounds [16 x float], [16 x float]* %9, i64 0, i64 %214
  %216 = load float, float* %215, align 4, !tbaa !32
  %217 = and i32 %206, 3840
  %218 = tail call fast float @air.convert.f.f32.s.i32(i32 %217) #6
  %219 = fmul fast float %218, %216
  %220 = extractelement <4 x float> %198, i64 1
  %221 = fadd fast float %219, %220
  %222 = insertelement <4 x float> %212, float %221, i64 1
  %223 = add nuw nsw i32 %199, 8
  %224 = zext i32 %223 to i64
  %225 = getelementptr inbounds [16 x float], [16 x float]* %9, i64 0, i64 %224
  %226 = load float, float* %225, align 4, !tbaa !32
  %227 = and i32 %206, 240
  %228 = tail call fast float @air.convert.f.f32.s.i32(i32 %227) #6
  %229 = fmul fast float %228, %226
  %230 = extractelement <4 x float> %198, i64 2
  %231 = fadd fast float %229, %230
  %232 = insertelement <4 x float> %222, float %231, i64 2
  %233 = add nuw nsw i32 %199, 9
  %234 = zext i32 %233 to i64
  %235 = getelementptr inbounds [16 x float], [16 x float]* %9, i64 0, i64 %234
  %236 = load float, float* %235, align 4, !tbaa !32
  %237 = and i32 %206, 61440
  %238 = tail call fast float @air.convert.f.f32.s.i32(i32 %237) #6
  %239 = fmul fast float %238, %236
  %240 = extractelement <4 x float> %198, i64 3
  %241 = fadd fast float %239, %240
  %242 = insertelement <4 x float> %232, float %241, i64 3
  %243 = getelementptr inbounds [16 x float], [16 x float]* %10, i64 0, i64 %200
  %244 = load float, float* %243, align 4, !tbaa !32
  %245 = getelementptr inbounds i16, i16 addrspace(1)* %126, i64 %203
  %246 = load i16, i16 addrspace(1)* %245, align 2, !tbaa !28, !alias.scope !43, !noalias !44
  %247 = zext i16 %246 to i32
  %248 = and i32 %247, 15
  %249 = tail call fast float @air.convert.f.f32.s.i32(i32 %248) #6
  %250 = fmul fast float %249, %244
  %251 = extractelement <4 x float> %197, i64 0
  %252 = fadd fast float %250, %251
  %253 = insertelement <4 x float> poison, float %252, i64 0
  %254 = getelementptr inbounds [16 x float], [16 x float]* %10, i64 0, i64 %214
  %255 = load float, float* %254, align 4, !tbaa !32
  %256 = and i32 %247, 3840
  %257 = tail call fast float @air.convert.f.f32.s.i32(i32 %256) #6
  %258 = fmul fast float %257, %255
  %259 = extractelement <4 x float> %197, i64 1
  %260 = fadd fast float %258, %259
  %261 = insertelement <4 x float> %253, float %260, i64 1
  %262 = getelementptr inbounds [16 x float], [16 x float]* %10, i64 0, i64 %224
  %263 = load float, float* %262, align 4, !tbaa !32
  %264 = and i32 %247, 240
  %265 = tail call fast float @air.convert.f.f32.s.i32(i32 %264) #6
  %266 = fmul fast float %265, %263
  %267 = extractelement <4 x float> %197, i64 2
  %268 = fadd fast float %266, %267
  %269 = insertelement <4 x float> %261, float %268, i64 2
  %270 = getelementptr inbounds [16 x float], [16 x float]* %10, i64 0, i64 %234
  %271 = load float, float* %270, align 4, !tbaa !32
  %272 = and i32 %247, 61440
  %273 = tail call fast float @air.convert.f.f32.s.i32(i32 %272) #6
  %274 = fmul fast float %273, %271
  %275 = extractelement <4 x float> %197, i64 3
  %276 = fadd fast float %274, %275
  %277 = insertelement <4 x float> %269, float %276, i64 3
  %278 = add nuw nsw i32 %196, 1
  %279 = icmp eq i32 %278, 4
  br i1 %279, label %127, label %195, !llvm.loop !48

280:                                              ; preds = %301, %51
  %281 = phi i1 [ true, %51 ], [ false, %301 ]
  %282 = phi i32 [ 0, %51 ], [ 1, %301 ]
  %283 = zext i32 %282 to i64
  %284 = getelementptr inbounds [2 x float], [2 x float]* %12, i64 0, i64 %283
  %285 = load float, float* %284, align 4, !tbaa !32
  %286 = tail call fast float @air.simd_sum.f32(float %285) #7
  br i1 %53, label %287, label %301

287:                                              ; preds = %280
  %288 = bitcast float %286 to i32
  %289 = and i32 %288, 2139095040
  %290 = icmp eq i32 %289, 2139095040
  %291 = lshr i32 %288, 16
  %292 = and i32 %291, 1
  %293 = add nuw nsw i32 %292, 32767
  %294 = select i1 %290, i32 0, i32 %293
  %295 = add i32 %294, %288
  %296 = and i32 %295, -65536
  %297 = or i32 %282, %19
  %298 = zext i32 %297 to i64
  %299 = getelementptr inbounds float, float addrspace(1)* %52, i64 %298
  %300 = bitcast float addrspace(1)* %299 to i32 addrspace(1)*
  store i32 %296, i32 addrspace(1)* %300, align 4, !tbaa !32, !alias.scope !50, !noalias !51
  br label %301

301:                                              ; preds = %287, %280
  br i1 %281, label %280, label %302, !llvm.loop !52

302:                                              ; preds = %301
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %23) #5
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %22) #5
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %21) #5
  ret void
}

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare float @air.simd_sum.f32(float) local_unnamed_addr #3

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare float @air.convert.f.f32.s.i32(i32) local_unnamed_addr #4

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn writeonly "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { convergent mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="96" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #2 = { argmemonly nocallback nofree nosync nounwind willreturn }
attributes #3 = { convergent mustprogress nounwind willreturn }
attributes #4 = { mustprogress nofree nosync nounwind readnone willreturn }
attributes #5 = { nounwind }
attributes #6 = { nounwind readnone willreturn }
attributes #7 = { convergent nounwind willreturn }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}
!air.version = !{!10}
!air.language_version = !{!11}
!air.compile_options = !{!12, !13, !14}
!air.kernel = !{!15}
!air.function_constants = !{!27}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 27, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{!"Apple metal version 32023.921 (metalfe-32023.921.6)"}
!10 = !{i32 2, i32 7, i32 0}
!11 = !{!"Metal", i32 3, i32 2, i32 0}
!12 = !{!"air.compile.denorms_disable"}
!13 = !{!"air.compile.fast_math_enable"}
!14 = !{!"air.compile.framebuffer_fetch_enable"}
!15 = !{void (%struct.ds4_metal_args_mul_mv addrspace(2)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(3)*, <3 x i32>, i16, i16)* @kernel_myds_q4_v41_qb_nb_bf16, !16, !17}
!16 = !{}
!17 = !{!18, !20, !21, !22, !23, !24, !25, !26}
!18 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 112, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !19, !"air.arg_type_size", i32 112, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"myds_q4_v41_args", !"air.arg_name", !"args", !"air.arg_unused"}
!19 = !{i32 0, i32 4, i32 0, !"int", !"ne00", i32 4, i32 4, i32 0, !"int", !"ne01", i32 8, i32 4, i32 0, !"int", !"ne02", i32 16, i32 8, i32 0, !"ulong", !"nb00", i32 24, i32 8, i32 0, !"ulong", !"nb01", i32 32, i32 8, i32 0, !"ulong", !"nb02", i32 40, i32 8, i32 0, !"ulong", !"nb03", i32 48, i32 4, i32 0, !"int", !"ne10", i32 52, i32 4, i32 0, !"int", !"ne11", i32 56, i32 4, i32 0, !"int", !"ne12", i32 64, i32 8, i32 0, !"ulong", !"nb10", i32 72, i32 8, i32 0, !"ulong", !"nb11", i32 80, i32 8, i32 0, !"ulong", !"nb12", i32 88, i32 8, i32 0, !"ulong", !"nb13", i32 96, i32 4, i32 0, !"int", !"ne0", i32 100, i32 4, i32 0, !"int", !"ne1", i32 104, i32 4, i32 0, !"int", !"nr0", i32 108, i32 2, i32 0, !"short", !"r2", i32 110, i32 2, i32 0, !"short", !"r3"}
!20 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"w"}
!21 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"x"}
!22 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"y"}
!23 = !{i32 4, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"scratch", !"air.arg_unused"}
!24 = !{i32 5, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"group"}
!25 = !{i32 6, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"lane"}
!26 = !{i32 7, !"air.simdgroup_index_in_threadgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"sg"}
!27 = !{i16 addrspace(2)* @_Z13FC_mul_mv_nsg.MTL_FC_INIT_600_s, !"short", !"FC_mul_mv_nsg", i32 600, i1 true}
!28 = !{!29, !29, i64 0}
!29 = !{!"short", !30, i64 0}
!30 = !{!"omnipotent char", !31, i64 0}
!31 = !{!"Simple C++ TBAA"}
!32 = !{!33, !33, i64 0}
!33 = !{!"float", !30, i64 0}
!34 = !{!35}
!35 = distinct !{!35, !36, !"air-alias-scope-arg(2)"}
!36 = distinct !{!36, !"air-alias-scopes(kernel_myds_q4_v41_qb_nb_bf16)"}
!37 = !{!38, !39}
!38 = distinct !{!38, !36, !"air-alias-scope-arg(1)"}
!39 = distinct !{!39, !36, !"air-alias-scope-arg(3)"}
!40 = distinct !{!40, !41}
!41 = !{!"llvm.loop.mustprogress"}
!42 = distinct !{!42, !41}
!43 = !{!38}
!44 = !{!35, !39}
!45 = !{!46, !46, i64 0}
!46 = !{!"half", !30, i64 0}
!47 = distinct !{!47, !41}
!48 = distinct !{!48, !41, !49}
!49 = !{!"llvm.loop.unroll.full"}
!50 = !{!39}
!51 = !{!38, !35}
!52 = distinct !{!52, !41}

