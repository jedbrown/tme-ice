#!/bin/sh

petscplot="python2.7 petscplot/petscplot --mode paper"

generate_figures() {
    mkdir -p figures
    ${petscplot} -t strong -m paper --events SNESSolve --stages 3 \
        shaheen/a/fast_strong_*_16_*.out : shaheen/a/fast_strong_*_32_4_*.out : shaheen/b/strong_tfs_* : shaheen/b/strong_hypre_vn_*_64_4_* : shaheen/b/strong_hypre_vn_*_64_2_* \
        --legend-labels 'Redundant 16:Redundant 32:TFS 32:BAMG iso 64:BAMG aniso 64+' --legend-loc 'lower left' \
        --title '' --xmin 4 \
        -o figures/shaheen-strong.pdf

    $petscplot -t weak -m paper --events SNESFunctionEval:SNESJacobianEval:PCSetUp:PCApply --stages 1 \
        shaheen/b/weak_hypre_1_53571.out shaheen/b/weak_hypre_4_53572.out shaheen/b/weak_hypre_32_53574.out shaheen/b/weak_hypre_256_try3_53754.out \
        --title '' --legend-loc 'upper right' --legend-labels 'Function Eval:Jacobian Eval:PC Setup:PC Apply'\
        -o figures/shaheen-weak.pdf

    $petscplot -t snes -m paper --solve-spec '1:' output/x.80km.m16p2l6.ew.log -o figures/x-80km-m16p2l6-ew.pdf

    $petscplot -t snes -m paper --solve-spec '1:' output/y.10km.m10p6l5.ew.log -o figures/y-10km-m10p6l5-ew.pdf

    $petscplot  -t algorithmic --width-pt 380 \
        output/z.m6p4l5.newton.icc.log : output/z.m6p4l5.picard.asm8.icc1.log : output/z.m6p4l6.picard.icc.log : output/z.m6p4l6.mult.n8.o0.r2.log \
        --legend-labels 'Newton, ICC(0), serial:Picard ASM(1)/ICC(1), 8 subdomains:Picard ICC(0) serial:Newton, V-cycle, 8 subdomains' \
        -o figures/linear4.pdf
}

case "$1" in
    figures)
        generate_figures
        ;;
    pdf)
        pdflatex hstat
        bibtex hstat
        pdflatex hstat
        pdflatex hstat
        ;;
    cover)
        pdflatex cover
        ;;
    submodules)
        # Warning: only intended to work for Jed's layout
        cp -rlTf ~/petscplot ./petscplot
        cp -rlTf ~/jedbib ./jedbib
        ;;
    *)
        echo "usage: $0 {figures|pdf|submodules}"
esac
