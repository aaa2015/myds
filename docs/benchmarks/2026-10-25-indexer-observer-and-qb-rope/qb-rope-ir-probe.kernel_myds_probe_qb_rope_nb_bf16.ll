source_filename = "kernel_myds_probe_qb_rope_nb_bf16"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v27-apple-macosx15.0.0"

%struct.ds4_metal_args_mul_mv = type { i32, i32, i32, i64, i64, i64, i64, i32, i32, i32, i64, i64, i64, i64, i32, i32, i32, i16, i16 }
%struct.ds4_metal_args_dsv41_rope = type { i32, i32, i32, i32, i32, i32, [32 x float] }

@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_all.metal, i8* null }]
@_ZL13FC_mul_mv_nsg = internal unnamed_addr addrspace(2) global i16 undef, align 2
@_Z13FC_mul_mv_nsg.MTL_FC_INIT_600_s = internal unnamed_addr addrspace(2) externally_initialized constant i16 undef, section "air.fc_initializer", align 2

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn writeonly
define internal void @_GLOBAL__sub_I_all.metal() #0 section "air.static_init" {
  %1 = load i16, i16 addrspace(2)* @_Z13FC_mul_mv_nsg.MTL_FC_INIT_600_s, align 2, !tbaa !30
  store i16 %1, i16 addrspace(2)* @_ZL13FC_mul_mv_nsg, align 2, !tbaa !30
  ret void
}

; Function Attrs: convergent mustprogress nounwind
define void @kernel_myds_probe_qb_rope_nb_bf16(%struct.ds4_metal_args_mul_mv addrspace(2)* nocapture noundef readnone align 8 dereferenceable(112) "air-buffer-no-alias" %0, i8 addrspace(1)* nocapture noundef readonly "air-buffer-no-alias" %1, i8 addrspace(1)* nocapture noundef readonly "air-buffer-no-alias" %2, i8 addrspace(1)* nocapture noundef writeonly "air-buffer-no-alias" %3, %struct.ds4_metal_args_dsv41_rope addrspace(2)* nocapture noundef readonly align 4 dereferenceable(152) "air-buffer-no-alias" %4, i8 addrspace(3)* nocapture noundef readnone "air-buffer-no-alias" %5, <3 x i32> noundef %6, i16 noundef %7, i16 noundef %8) local_unnamed_addr #1 {
  %10 = alloca [16 x float], align 4
  %11 = alloca [16 x float], align 4
  %12 = alloca i64, align 8
  %13 = bitcast i64* %12 to [2 x float]*
  %14 = load i16, i16 addrspace(2)* @_ZL13FC_mul_mv_nsg, align 2, !tbaa !30
  %15 = extractelement <3 x i32> %6, i64 0
  %16 = sext i16 %14 to i32
  %17 = mul i32 %15, %16
  %18 = zext i16 %8 to i32
  %19 = add i32 %17, %18
  %20 = shl i32 %19, 1
  %21 = mul i32 %19, 1440
  %22 = bitcast [16 x float]* %10 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %22) #5
  %23 = bitcast [16 x float]* %11 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %23) #5
  %24 = bitcast i64* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %24) #5
  store i64 0, i64* %12, align 8
  %25 = icmp ult i16 %7, 40
  br i1 %25, label %26, label %57

26:                                               ; preds = %9
  %27 = lshr i16 %7, 3
  %28 = zext i16 %27 to i32
  %29 = bitcast i8 addrspace(1)* %2 to float addrspace(1)*
  %30 = shl nuw nsw i32 %28, 8
  %31 = zext i32 %30 to i64
  %32 = getelementptr inbounds float, float addrspace(1)* %29, i64 %31
  %33 = lshr i16 %7, 2
  %34 = and i16 %33, 1
  %35 = zext i16 %34 to i32
  %36 = shl nuw nsw i32 %35, 6
  %37 = zext i32 %36 to i64
  %38 = getelementptr inbounds float, float addrspace(1)* %32, i64 %37
  %39 = and i16 %7, 3
  %40 = zext i16 %39 to i32
  %41 = shl nuw nsw i32 %40, 3
  %42 = zext i32 %41 to i64
  %43 = getelementptr inbounds float, float addrspace(1)* %38, i64 %42
  %44 = zext i16 %34 to i64
  %45 = shl nuw nsw i32 %35, 4
  %46 = zext i32 %45 to i64
  %47 = shl nuw nsw i32 %40, 2
  %48 = zext i32 %47 to i64
  br label %49

49:                                               ; preds = %115, %26
  %50 = phi float addrspace(1)* [ %43, %26 ], [ %116, %115 ]
  %51 = phi i32 [ %28, %26 ], [ 4, %115 ]
  br label %77

52:                                               ; preds = %115
  %53 = bitcast i64* %12 to float*
  %54 = load float, float* %53, align 8, !tbaa !34
  %55 = getelementptr inbounds [2 x float], [2 x float]* %13, i64 0, i64 1
  %56 = load float, float* %55, align 4, !tbaa !34
  br label %57

57:                                               ; preds = %52, %9
  %58 = phi float [ %56, %52 ], [ 0.000000e+00, %9 ]
  %59 = phi float [ %54, %52 ], [ 0.000000e+00, %9 ]
  %60 = bitcast i8 addrspace(1)* %3 to float addrspace(1)*
  %61 = tail call fast float @air.simd_sum.f32(float %59) #6
  %62 = tail call fast float @air.simd_sum.f32(float %58) #6
  %63 = icmp eq i16 %7, 0
  br i1 %63, label %290, label %364

64:                                               ; preds = %77
  %65 = mul nuw nsw i32 %51, 144
  %66 = add i32 %65, %21
  %67 = zext i32 %66 to i64
  %68 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %67
  %69 = getelementptr inbounds i8, i8 addrspace(1)* %68, i64 4
  %70 = bitcast i8 addrspace(1)* %69 to i16 addrspace(1)*
  %71 = getelementptr inbounds i16, i16 addrspace(1)* %70, i64 %44
  %72 = getelementptr inbounds i8, i8 addrspace(1)* %68, i64 16
  %73 = bitcast i8 addrspace(1)* %72 to i16 addrspace(1)*
  %74 = getelementptr inbounds i16, i16 addrspace(1)* %73, i64 %46
  %75 = getelementptr inbounds i16, i16 addrspace(1)* %74, i64 %48
  %76 = bitcast i8 addrspace(1)* %68 to half addrspace(1)*
  br label %118

77:                                               ; preds = %77, %49
  %78 = phi i32 [ 0, %49 ], [ %113, %77 ]
  %79 = phi <4 x float> [ zeroinitializer, %49 ], [ %112, %77 ]
  %80 = zext i32 %78 to i64
  %81 = getelementptr inbounds float, float addrspace(1)* %50, i64 %80
  %82 = load float, float addrspace(1)* %81, align 4, !tbaa !34, !alias.scope !36, !noalias !39
  %83 = getelementptr inbounds [16 x float], [16 x float]* %10, i64 0, i64 %80
  store float %82, float* %83, align 4, !tbaa !34
  %84 = extractelement <4 x float> %79, i64 0
  %85 = fadd fast float %84, %82
  %86 = insertelement <4 x float> poison, float %85, i64 0
  %87 = add nuw nsw i32 %78, 32
  %88 = zext i32 %87 to i64
  %89 = getelementptr inbounds float, float addrspace(1)* %50, i64 %88
  %90 = load float, float addrspace(1)* %89, align 4, !tbaa !34, !alias.scope !36, !noalias !39
  %91 = add nuw nsw i32 %78, 8
  %92 = zext i32 %91 to i64
  %93 = getelementptr inbounds [16 x float], [16 x float]* %10, i64 0, i64 %92
  store float %90, float* %93, align 4, !tbaa !34
  %94 = extractelement <4 x float> %79, i64 1
  %95 = fadd fast float %90, %94
  %96 = insertelement <4 x float> %86, float %95, i64 1
  %97 = add nuw nsw i32 %78, 128
  %98 = zext i32 %97 to i64
  %99 = getelementptr inbounds float, float addrspace(1)* %50, i64 %98
  %100 = load float, float addrspace(1)* %99, align 4, !tbaa !34, !alias.scope !36, !noalias !39
  %101 = getelementptr inbounds [16 x float], [16 x float]* %11, i64 0, i64 %80
  store float %100, float* %101, align 4, !tbaa !34
  %102 = extractelement <4 x float> %79, i64 2
  %103 = fadd fast float %100, %102
  %104 = insertelement <4 x float> %96, float %103, i64 2
  %105 = add nuw nsw i32 %78, 160
  %106 = zext i32 %105 to i64
  %107 = getelementptr inbounds float, float addrspace(1)* %50, i64 %106
  %108 = load float, float addrspace(1)* %107, align 4, !tbaa !34, !alias.scope !36, !noalias !39
  %109 = getelementptr inbounds [16 x float], [16 x float]* %11, i64 0, i64 %92
  store float %108, float* %109, align 4, !tbaa !34
  %110 = extractelement <4 x float> %79, i64 3
  %111 = fadd fast float %108, %110
  %112 = insertelement <4 x float> %104, float %111, i64 3
  %113 = add nuw nsw i32 %78, 1
  %114 = icmp eq i32 %113, 8
  br i1 %114, label %64, label %77, !llvm.loop !43

115:                                              ; preds = %137
  %116 = getelementptr inbounds float, float addrspace(1)* %50, i64 1024
  %117 = icmp eq i32 %51, 0
  br i1 %117, label %49, label %52, !llvm.loop !45

118:                                              ; preds = %137, %64
  %119 = phi i16 addrspace(1)* [ %71, %64 ], [ %201, %137 ]
  %120 = phi i16 addrspace(1)* [ %75, %64 ], [ %200, %137 ]
  %121 = phi half addrspace(1)* [ %76, %64 ], [ %202, %137 ]
  %122 = phi i16 [ 0, %64 ], [ %203, %137 ]
  %123 = load i16, i16 addrspace(1)* %119, align 2, !tbaa !30, !alias.scope !46, !noalias !47
  %124 = lshr i16 %123, 8
  %125 = getelementptr inbounds i16, i16 addrspace(1)* %119, i64 2
  %126 = load i16, i16 addrspace(1)* %125, align 2, !tbaa !30, !alias.scope !46, !noalias !47
  %127 = lshr i16 %126, 8
  %128 = getelementptr inbounds i16, i16 addrspace(1)* %119, i64 4
  %129 = load i16, i16 addrspace(1)* %128, align 2, !tbaa !30, !alias.scope !46, !noalias !47
  %130 = and i16 %129, 3855
  %131 = lshr i16 %123, 2
  %132 = and i16 %131, 12336
  %133 = or i16 %130, %132
  %134 = lshr i16 %133, 8
  %135 = lshr i16 %126, 2
  %136 = getelementptr inbounds i16, i16 addrspace(1)* %120, i64 32
  br label %205

137:                                              ; preds = %205
  %138 = and i16 %124, 63
  %139 = and i16 %127, 63
  %140 = lshr i16 %129, 4
  %141 = and i16 %135, 12336
  %142 = zext i16 %138 to i32
  %143 = zext i16 %139 to i32
  %144 = zext i16 %134 to i32
  %145 = and i16 %140, 3855
  %146 = or i16 %145, %141
  %147 = load half, half addrspace(1)* %121, align 2, !tbaa !48, !alias.scope !46, !noalias !47
  %148 = fpext half %147 to float
  %149 = fmul fast float %231, 3.906250e-03
  %150 = fadd fast float %149, %221
  %151 = and i16 %123, 63
  %152 = zext i16 %151 to i32
  %153 = tail call fast float @air.convert.f.f32.s.i32(i32 %152) #7
  %154 = fmul fast float %153, %150
  %155 = fmul fast float %251, 3.906250e-03
  %156 = fadd fast float %155, %241
  %157 = tail call fast float @air.convert.f.f32.s.i32(i32 %142) #7
  %158 = fmul fast float %157, %156
  %159 = fmul fast float %270, 3.906250e-03
  %160 = fadd fast float %159, %262
  %161 = and i16 %133, 63
  %162 = zext i16 %161 to i32
  %163 = tail call fast float @air.convert.f.f32.s.i32(i32 %162) #7
  %164 = fmul fast float %163, %160
  %165 = fmul fast float %286, 3.906250e-03
  %166 = fadd fast float %165, %278
  %167 = tail call fast float @air.convert.f.f32.s.i32(i32 %144) #7
  %168 = fmul fast float %167, %166
  %169 = fadd fast float %168, %158
  %170 = fmul fast float %169, 6.250000e-02
  %171 = fadd fast float %164, %154
  %172 = fadd fast float %171, %170
  %173 = fmul fast float %172, %148
  %174 = getelementptr inbounds half, half addrspace(1)* %121, i64 1
  %175 = load half, half addrspace(1)* %174, align 2, !tbaa !48, !alias.scope !46, !noalias !47
  %176 = fpext half %175 to float
  %177 = and i16 %126, 63
  %178 = zext i16 %177 to i32
  %179 = tail call fast float @air.convert.f.f32.s.i32(i32 %178) #7
  %180 = fmul fast float %179, %85
  %181 = tail call fast float @air.convert.f.f32.s.i32(i32 %143) #7
  %182 = fmul fast float %181, %95
  %183 = fadd fast float %182, %180
  %184 = and i16 %146, 63
  %185 = zext i16 %184 to i32
  %186 = tail call fast float @air.convert.f.f32.s.i32(i32 %185) #7
  %187 = fmul fast float %186, %103
  %188 = fadd fast float %183, %187
  %189 = lshr i16 %146, 8
  %190 = zext i16 %189 to i32
  %191 = tail call fast float @air.convert.f.f32.s.i32(i32 %190) #7
  %192 = fmul fast float %191, %111
  %193 = fadd fast float %188, %192
  %194 = zext i16 %122 to i64
  %195 = getelementptr inbounds [2 x float], [2 x float]* %13, i64 0, i64 %194
  %196 = load float, float* %195, align 4, !tbaa !34
  %197 = fadd fast float %196, %173
  %198 = fmul fast float %193, %176
  %199 = fsub fast float %197, %198
  store float %199, float* %195, align 4, !tbaa !34
  %200 = getelementptr inbounds i16, i16 addrspace(1)* %120, i64 360
  %201 = getelementptr inbounds i16, i16 addrspace(1)* %119, i64 360
  %202 = getelementptr inbounds half, half addrspace(1)* %121, i64 360
  %203 = add nuw nsw i16 %122, 1
  %204 = icmp eq i16 %203, 2
  br i1 %204, label %115, label %118, !llvm.loop !50

205:                                              ; preds = %205, %118
  %206 = phi i32 [ 0, %118 ], [ %288, %205 ]
  %207 = phi <4 x float> [ zeroinitializer, %118 ], [ %252, %205 ]
  %208 = phi <4 x float> [ zeroinitializer, %118 ], [ %287, %205 ]
  %209 = shl nuw i32 %206, 1
  %210 = zext i32 %209 to i64
  %211 = getelementptr inbounds [16 x float], [16 x float]* %10, i64 0, i64 %210
  %212 = load float, float* %211, align 4, !tbaa !34
  %213 = zext i32 %206 to i64
  %214 = getelementptr inbounds i16, i16 addrspace(1)* %120, i64 %213
  %215 = load i16, i16 addrspace(1)* %214, align 2, !tbaa !30, !alias.scope !46, !noalias !47
  %216 = zext i16 %215 to i32
  %217 = and i32 %216, 15
  %218 = tail call fast float @air.convert.f.f32.s.i32(i32 %217) #7
  %219 = fmul fast float %218, %212
  %220 = extractelement <4 x float> %207, i64 0
  %221 = fadd fast float %219, %220
  %222 = insertelement <4 x float> poison, float %221, i64 0
  %223 = or i32 %209, 1
  %224 = zext i32 %223 to i64
  %225 = getelementptr inbounds [16 x float], [16 x float]* %10, i64 0, i64 %224
  %226 = load float, float* %225, align 4, !tbaa !34
  %227 = and i32 %216, 3840
  %228 = tail call fast float @air.convert.f.f32.s.i32(i32 %227) #7
  %229 = fmul fast float %228, %226
  %230 = extractelement <4 x float> %207, i64 1
  %231 = fadd fast float %229, %230
  %232 = insertelement <4 x float> %222, float %231, i64 1
  %233 = add nuw nsw i32 %209, 8
  %234 = zext i32 %233 to i64
  %235 = getelementptr inbounds [16 x float], [16 x float]* %10, i64 0, i64 %234
  %236 = load float, float* %235, align 4, !tbaa !34
  %237 = and i32 %216, 240
  %238 = tail call fast float @air.convert.f.f32.s.i32(i32 %237) #7
  %239 = fmul fast float %238, %236
  %240 = extractelement <4 x float> %207, i64 2
  %241 = fadd fast float %239, %240
  %242 = insertelement <4 x float> %232, float %241, i64 2
  %243 = add nuw nsw i32 %209, 9
  %244 = zext i32 %243 to i64
  %245 = getelementptr inbounds [16 x float], [16 x float]* %10, i64 0, i64 %244
  %246 = load float, float* %245, align 4, !tbaa !34
  %247 = and i32 %216, 61440
  %248 = tail call fast float @air.convert.f.f32.s.i32(i32 %247) #7
  %249 = fmul fast float %248, %246
  %250 = extractelement <4 x float> %207, i64 3
  %251 = fadd fast float %249, %250
  %252 = insertelement <4 x float> %242, float %251, i64 3
  %253 = getelementptr inbounds [16 x float], [16 x float]* %11, i64 0, i64 %210
  %254 = load float, float* %253, align 4, !tbaa !34
  %255 = getelementptr inbounds i16, i16 addrspace(1)* %136, i64 %213
  %256 = load i16, i16 addrspace(1)* %255, align 2, !tbaa !30, !alias.scope !46, !noalias !47
  %257 = zext i16 %256 to i32
  %258 = and i32 %257, 15
  %259 = tail call fast float @air.convert.f.f32.s.i32(i32 %258) #7
  %260 = fmul fast float %259, %254
  %261 = extractelement <4 x float> %208, i64 0
  %262 = fadd fast float %260, %261
  %263 = insertelement <4 x float> poison, float %262, i64 0
  %264 = getelementptr inbounds [16 x float], [16 x float]* %11, i64 0, i64 %224
  %265 = load float, float* %264, align 4, !tbaa !34
  %266 = and i32 %257, 3840
  %267 = tail call fast float @air.convert.f.f32.s.i32(i32 %266) #7
  %268 = fmul fast float %267, %265
  %269 = extractelement <4 x float> %208, i64 1
  %270 = fadd fast float %268, %269
  %271 = insertelement <4 x float> %263, float %270, i64 1
  %272 = getelementptr inbounds [16 x float], [16 x float]* %11, i64 0, i64 %234
  %273 = load float, float* %272, align 4, !tbaa !34
  %274 = and i32 %257, 240
  %275 = tail call fast float @air.convert.f.f32.s.i32(i32 %274) #7
  %276 = fmul fast float %275, %273
  %277 = extractelement <4 x float> %208, i64 2
  %278 = fadd fast float %276, %277
  %279 = insertelement <4 x float> %271, float %278, i64 2
  %280 = getelementptr inbounds [16 x float], [16 x float]* %11, i64 0, i64 %244
  %281 = load float, float* %280, align 4, !tbaa !34
  %282 = and i32 %257, 61440
  %283 = tail call fast float @air.convert.f.f32.s.i32(i32 %282) #7
  %284 = fmul fast float %283, %281
  %285 = extractelement <4 x float> %208, i64 3
  %286 = fadd fast float %284, %285
  %287 = insertelement <4 x float> %279, float %286, i64 3
  %288 = add nuw nsw i32 %206, 1
  %289 = icmp eq i32 %288, 4
  br i1 %289, label %137, label %205, !llvm.loop !51

290:                                              ; preds = %57
  %291 = bitcast float %61 to i32
  %292 = and i32 %291, 2139095040
  %293 = icmp eq i32 %292, 2139095040
  %294 = lshr i32 %291, 16
  %295 = and i32 %294, 1
  %296 = add nuw nsw i32 %295, 32767
  %297 = select i1 %293, i32 0, i32 %296
  %298 = add i32 %297, %291
  %299 = and i32 %298, -65536
  %300 = bitcast float %62 to i32
  %301 = and i32 %300, 2139095040
  %302 = icmp eq i32 %301, 2139095040
  %303 = lshr i32 %300, 16
  %304 = and i32 %303, 1
  %305 = add nuw nsw i32 %304, 32767
  %306 = select i1 %302, i32 0, i32 %305
  %307 = add i32 %306, %300
  %308 = and i32 %307, -65536
  %309 = and i32 %20, 510
  %310 = icmp ugt i32 %309, 447
  br i1 %310, label %311, label %356

311:                                              ; preds = %290
  %312 = bitcast i32 %308 to float
  %313 = bitcast i32 %299 to float
  %314 = add nsw i32 %309, -448
  %315 = lshr exact i32 %314, 1
  %316 = getelementptr inbounds %struct.ds4_metal_args_dsv41_rope, %struct.ds4_metal_args_dsv41_rope addrspace(2)* %4, i64 0, i32 3
  %317 = load i32, i32 addrspace(2)* %316, align 4, !tbaa !53, !alias.scope !56, !noalias !57
  %318 = tail call fast float @air.convert.f.f32.u.i32(i32 %317) #7
  %319 = zext i32 %315 to i64
  %320 = getelementptr inbounds %struct.ds4_metal_args_dsv41_rope, %struct.ds4_metal_args_dsv41_rope addrspace(2)* %4, i64 0, i32 6, i64 %319
  %321 = load float, float addrspace(2)* %320, align 4, !tbaa !34, !alias.scope !56, !noalias !57
  %322 = fmul fast float %321, %318
  %323 = tail call fast float @air.cos.f32(float %322) #7
  %324 = tail call fast float @air.sin.f32(float %322) #7
  %325 = fmul fast float %323, %313
  %326 = fmul fast float %324, %312
  %327 = fsub fast float %325, %326
  %328 = bitcast float %327 to i32
  %329 = and i32 %328, 2139095040
  %330 = icmp eq i32 %329, 2139095040
  %331 = lshr i32 %328, 16
  %332 = and i32 %331, 1
  %333 = add nuw nsw i32 %332, 32767
  %334 = select i1 %330, i32 0, i32 %333
  %335 = add i32 %334, %328
  %336 = and i32 %335, -65536
  %337 = zext i32 %20 to i64
  %338 = getelementptr inbounds float, float addrspace(1)* %60, i64 %337
  %339 = bitcast float addrspace(1)* %338 to i32 addrspace(1)*
  store i32 %336, i32 addrspace(1)* %339, align 4, !tbaa !34, !alias.scope !58, !noalias !59
  %340 = fmul fast float %324, %313
  %341 = fmul fast float %323, %312
  %342 = fadd fast float %340, %341
  %343 = bitcast float %342 to i32
  %344 = and i32 %343, 2139095040
  %345 = icmp eq i32 %344, 2139095040
  %346 = lshr i32 %343, 16
  %347 = and i32 %346, 1
  %348 = add nuw nsw i32 %347, 32767
  %349 = select i1 %345, i32 0, i32 %348
  %350 = add i32 %349, %343
  %351 = and i32 %350, -65536
  %352 = or i32 %20, 1
  %353 = zext i32 %352 to i64
  %354 = getelementptr inbounds float, float addrspace(1)* %60, i64 %353
  %355 = bitcast float addrspace(1)* %354 to i32 addrspace(1)*
  store i32 %351, i32 addrspace(1)* %355, align 4, !tbaa !34, !alias.scope !58, !noalias !59
  br label %364

356:                                              ; preds = %290
  %357 = zext i32 %20 to i64
  %358 = getelementptr inbounds float, float addrspace(1)* %60, i64 %357
  %359 = bitcast float addrspace(1)* %358 to i32 addrspace(1)*
  store i32 %299, i32 addrspace(1)* %359, align 4, !tbaa !34, !alias.scope !58, !noalias !59
  %360 = or i32 %20, 1
  %361 = zext i32 %360 to i64
  %362 = getelementptr inbounds float, float addrspace(1)* %60, i64 %361
  %363 = bitcast float addrspace(1)* %362 to i32 addrspace(1)*
  store i32 %308, i32 addrspace(1)* %363, align 4, !tbaa !34, !alias.scope !58, !noalias !59
  br label %364

364:                                              ; preds = %356, %311, %57
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %24) #5
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %23) #5
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %22) #5
  ret void
}

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #2

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare float @air.sin.f32(float) local_unnamed_addr #3

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare float @air.cos.f32(float) local_unnamed_addr #3

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr #3

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare float @air.convert.f.f32.s.i32(i32) local_unnamed_addr #3

; Function Attrs: convergent mustprogress nounwind willreturn
declare float @air.simd_sum.f32(float) local_unnamed_addr #4

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn writeonly "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { convergent mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="96" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #2 = { argmemonly nocallback nofree nosync nounwind willreturn }
attributes #3 = { mustprogress nofree nosync nounwind readnone willreturn }
attributes #4 = { convergent mustprogress nounwind willreturn }
attributes #5 = { nounwind }
attributes #6 = { convergent nounwind willreturn }
attributes #7 = { nounwind readnone willreturn }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}
!air.version = !{!10}
!air.language_version = !{!11}
!air.compile_options = !{!12, !13, !14}
!air.kernel = !{!15}
!air.function_constants = !{!29}

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
!15 = !{void (%struct.ds4_metal_args_mul_mv addrspace(2)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, %struct.ds4_metal_args_dsv41_rope addrspace(2)*, i8 addrspace(3)*, <3 x i32>, i16, i16)* @kernel_myds_probe_qb_rope_nb_bf16, !16, !17}
!16 = !{}
!17 = !{!18, !20, !21, !22, !23, !25, !26, !27, !28}
!18 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 112, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !19, !"air.arg_type_size", i32 112, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"myds_q4_v41_args", !"air.arg_name", !"args", !"air.arg_unused"}
!19 = !{i32 0, i32 4, i32 0, !"int", !"ne00", i32 4, i32 4, i32 0, !"int", !"ne01", i32 8, i32 4, i32 0, !"int", !"ne02", i32 16, i32 8, i32 0, !"ulong", !"nb00", i32 24, i32 8, i32 0, !"ulong", !"nb01", i32 32, i32 8, i32 0, !"ulong", !"nb02", i32 40, i32 8, i32 0, !"ulong", !"nb03", i32 48, i32 4, i32 0, !"int", !"ne10", i32 52, i32 4, i32 0, !"int", !"ne11", i32 56, i32 4, i32 0, !"int", !"ne12", i32 64, i32 8, i32 0, !"ulong", !"nb10", i32 72, i32 8, i32 0, !"ulong", !"nb11", i32 80, i32 8, i32 0, !"ulong", !"nb12", i32 88, i32 8, i32 0, !"ulong", !"nb13", i32 96, i32 4, i32 0, !"int", !"ne0", i32 100, i32 4, i32 0, !"int", !"ne1", i32 104, i32 4, i32 0, !"int", !"nr0", i32 108, i32 2, i32 0, !"short", !"r2", i32 110, i32 2, i32 0, !"short", !"r3"}
!20 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"w"}
!21 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"x"}
!22 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"y"}
!23 = !{i32 4, !"air.buffer", !"air.buffer_size", i32 152, !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !24, !"air.arg_type_size", i32 152, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"ds4_metal_args_dsv41_rope", !"air.arg_name", !"rope"}
!24 = !{i32 0, i32 4, i32 0, !"uint", !"width", i32 4, i32 4, i32 0, !"uint", !"heads", i32 8, i32 4, i32 0, !"uint", !"rows", i32 12, i32 4, i32 0, !"uint", !"start", i32 16, i32 4, i32 0, !"uint", !"inverse", i32 20, i32 4, i32 0, !"uint", !"stride", i32 24, i32 4, i32 32, !"float", !"frequencies"}
!25 = !{i32 5, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"scratch", !"air.arg_unused"}
!26 = !{i32 6, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"group"}
!27 = !{i32 7, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"lane"}
!28 = !{i32 8, !"air.simdgroup_index_in_threadgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"sg"}
!29 = !{i16 addrspace(2)* @_Z13FC_mul_mv_nsg.MTL_FC_INIT_600_s, !"short", !"FC_mul_mv_nsg", i32 600, i1 true}
!30 = !{!31, !31, i64 0}
!31 = !{!"short", !32, i64 0}
!32 = !{!"omnipotent char", !33, i64 0}
!33 = !{!"Simple C++ TBAA"}
!34 = !{!35, !35, i64 0}
!35 = !{!"float", !32, i64 0}
!36 = !{!37}
!37 = distinct !{!37, !38, !"air-alias-scope-arg(2)"}
!38 = distinct !{!38, !"air-alias-scopes(kernel_myds_probe_qb_rope_nb_bf16)"}
!39 = !{!40, !41, !42}
!40 = distinct !{!40, !38, !"air-alias-scope-arg(1)"}
!41 = distinct !{!41, !38, !"air-alias-scope-arg(3)"}
!42 = distinct !{!42, !38, !"air-alias-scope-arg(4)"}
!43 = distinct !{!43, !44}
!44 = !{!"llvm.loop.mustprogress"}
!45 = distinct !{!45, !44}
!46 = !{!40}
!47 = !{!37, !41, !42}
!48 = !{!49, !49, i64 0}
!49 = !{!"half", !32, i64 0}
!50 = distinct !{!50, !44}
!51 = distinct !{!51, !44, !52}
!52 = !{!"llvm.loop.unroll.full"}
!53 = !{!54, !55, i64 12}
!54 = !{!"_ZTS25ds4_metal_args_dsv41_rope", !55, i64 0, !55, i64 4, !55, i64 8, !55, i64 12, !55, i64 16, !55, i64 20, !32, i64 24}
!55 = !{!"int", !32, i64 0}
!56 = !{!42}
!57 = !{!40, !37, !41}
!58 = !{!41}
!59 = !{!40, !37, !42}
