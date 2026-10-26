source_filename = "kernel_dsv41_rope"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v27-apple-macosx15.0.0"

%struct.ds4_metal_args_dsv41_rope = type { i32, i32, i32, i32, i32, i32, [32 x float] }

@llvm.global_ctors = appending global [0 x { i32, void ()*, i8* }] zeroinitializer

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
define void @kernel_dsv41_rope(%struct.ds4_metal_args_dsv41_rope addrspace(2)* nocapture noundef readonly align 4 dereferenceable(152) "air-buffer-no-alias" %0, float addrspace(1)* nocapture noundef "air-buffer-no-alias" %1, <2 x i32> noundef %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = getelementptr inbounds %struct.ds4_metal_args_dsv41_rope, %struct.ds4_metal_args_dsv41_rope addrspace(2)* %0, i64 0, i32 3
  %6 = load i32, i32 addrspace(2)* %5, align 4, !tbaa !23, !alias.scope !28, !noalias !31
  %7 = extractelement <2 x i32> %2, i64 1
  %8 = getelementptr inbounds %struct.ds4_metal_args_dsv41_rope, %struct.ds4_metal_args_dsv41_rope addrspace(2)* %0, i64 0, i32 5
  %9 = load i32, i32 addrspace(2)* %8, align 4, !tbaa !33, !alias.scope !28, !noalias !31
  %10 = mul i32 %9, %7
  %11 = add i32 %10, %6
  %12 = tail call fast float @air.convert.f.f32.u.i32(i32 %11) #2
  %13 = zext i32 %3 to i64
  %14 = getelementptr inbounds %struct.ds4_metal_args_dsv41_rope, %struct.ds4_metal_args_dsv41_rope addrspace(2)* %0, i64 0, i32 6, i64 %13
  %15 = load float, float addrspace(2)* %14, align 4, !tbaa !34, !alias.scope !28, !noalias !31
  %16 = fmul fast float %15, %12
  %17 = getelementptr inbounds %struct.ds4_metal_args_dsv41_rope, %struct.ds4_metal_args_dsv41_rope addrspace(2)* %0, i64 0, i32 4
  %18 = load i32, i32 addrspace(2)* %17, align 4, !tbaa !36, !alias.scope !28, !noalias !31
  %19 = icmp eq i32 %18, 0
  %20 = tail call fast float @air.sin.f32(float %16) #2
  %21 = fneg fast float %20
  %22 = select fast i1 %19, float %20, float %21
  %23 = tail call fast float @air.cos.f32(float %16) #2
  %24 = zext i32 %7 to i64
  %25 = getelementptr inbounds %struct.ds4_metal_args_dsv41_rope, %struct.ds4_metal_args_dsv41_rope addrspace(2)* %0, i64 0, i32 1
  %26 = load i32, i32 addrspace(2)* %25, align 4, !tbaa !37, !alias.scope !28, !noalias !31
  %27 = zext i32 %26 to i64
  %28 = mul nuw i64 %27, %24
  %29 = extractelement <2 x i32> %2, i64 0
  %30 = zext i32 %29 to i64
  %31 = add nuw i64 %28, %30
  %32 = getelementptr inbounds %struct.ds4_metal_args_dsv41_rope, %struct.ds4_metal_args_dsv41_rope addrspace(2)* %0, i64 0, i32 0
  %33 = load i32, i32 addrspace(2)* %32, align 4, !tbaa !38, !alias.scope !28, !noalias !31
  %34 = zext i32 %33 to i64
  %35 = mul i64 %31, %34
  %36 = shl i32 %3, 1
  %37 = zext i32 %36 to i64
  %38 = add nsw i64 %37, -64
  %39 = add nsw i64 %38, %34
  %40 = add i64 %39, %35
  %41 = getelementptr inbounds float, float addrspace(1)* %1, i64 %40
  %42 = load float, float addrspace(1)* %41, align 4, !tbaa !34, !alias.scope !31, !noalias !28
  %43 = add i64 %40, 1
  %44 = getelementptr inbounds float, float addrspace(1)* %1, i64 %43
  %45 = load float, float addrspace(1)* %44, align 4, !tbaa !34, !alias.scope !31, !noalias !28
  %46 = fmul fast float %42, %23
  %47 = fmul fast float %45, %22
  %48 = fsub fast float %46, %47
  %49 = bitcast float %48 to i32
  %50 = and i32 %49, 2139095040
  %51 = icmp eq i32 %50, 2139095040
  %52 = lshr i32 %49, 16
  %53 = and i32 %52, 1
  %54 = add nuw nsw i32 %53, 32767
  %55 = select i1 %51, i32 0, i32 %54
  %56 = add i32 %55, %49
  %57 = and i32 %56, -65536
  %58 = bitcast float addrspace(1)* %41 to i32 addrspace(1)*
  store i32 %57, i32 addrspace(1)* %58, align 4, !tbaa !34, !alias.scope !31, !noalias !28
  %59 = fmul fast float %42, %22
  %60 = fmul fast float %45, %23
  %61 = fadd fast float %60, %59
  %62 = bitcast float %61 to i32
  %63 = and i32 %62, 2139095040
  %64 = icmp eq i32 %63, 2139095040
  %65 = lshr i32 %62, 16
  %66 = and i32 %65, 1
  %67 = add nuw nsw i32 %66, 32767
  %68 = select i1 %64, i32 0, i32 %67
  %69 = add i32 %68, %62
  %70 = and i32 %69, -65536
  %71 = bitcast float addrspace(1)* %44 to i32 addrspace(1)*
  store i32 %70, i32 addrspace(1)* %71, align 4, !tbaa !34, !alias.scope !31, !noalias !28
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare float @air.cos.f32(float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare float @air.sin.f32(float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr #1

attributes #0 = { argmemonly mustprogress nofree nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="64" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind readnone willreturn }
attributes #2 = { nounwind readnone willreturn }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}
!air.version = !{!10}
!air.language_version = !{!11}
!air.compile_options = !{!12, !13, !14}
!air.kernel = !{!15}

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
!15 = !{void (%struct.ds4_metal_args_dsv41_rope addrspace(2)*, float addrspace(1)*, <2 x i32>, i32)* @kernel_dsv41_rope, !16, !17}
!16 = !{}
!17 = !{!18, !20, !21, !22}
!18 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 152, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !19, !"air.arg_type_size", i32 152, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"ds4_metal_args_dsv41_rope", !"air.arg_name", !"args"}
!19 = !{i32 0, i32 4, i32 0, !"uint", !"width", i32 4, i32 4, i32 0, !"uint", !"heads", i32 8, i32 4, i32 0, !"uint", !"rows", i32 12, i32 4, i32 0, !"uint", !"start", i32 16, i32 4, i32 0, !"uint", !"inverse", i32 20, i32 4, i32 0, !"uint", !"stride", i32 24, i32 4, i32 32, !"float", !"frequencies"}
!20 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"x"}
!21 = !{i32 2, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"group"}
!22 = !{i32 3, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"lane"}
!23 = !{!24, !25, i64 12}
!24 = !{!"_ZTS25ds4_metal_args_dsv41_rope", !25, i64 0, !25, i64 4, !25, i64 8, !25, i64 12, !25, i64 16, !25, i64 20, !26, i64 24}
!25 = !{!"int", !26, i64 0}
!26 = !{!"omnipotent char", !27, i64 0}
!27 = !{!"Simple C++ TBAA"}
!28 = !{!29}
!29 = distinct !{!29, !30, !"air-alias-scope-arg(0)"}
!30 = distinct !{!30, !"air-alias-scopes(kernel_dsv41_rope)"}
!31 = !{!32}
!32 = distinct !{!32, !30, !"air-alias-scope-arg(1)"}
!33 = !{!24, !25, i64 20}
!34 = !{!35, !35, i64 0}
!35 = !{!"float", !26, i64 0}
!36 = !{!24, !25, i64 16}
!37 = !{!24, !25, i64 4}
!38 = !{!24, !25, i64 0}

