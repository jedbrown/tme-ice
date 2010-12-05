#!/bin/sh

petscplot="python2.7 petscplot/petscplot --mode paper"

generate_figures() {
    mkdir -p figures
    ${petscplot} -t strong --events SNESSolve --stages 3 \
        shaheen/fast_strong_*_16_4_*.out : shaheen/fast_strong_*_32_4_*.out : shaheen/strong_tfs_*.out \
        --legend-labels '$256\times 256\times 48$ Redundant:$512\times 512\times 48$ Redundant:$512\times 512\times 48$ TFS' \
        -o figures/shaheen-strong.pdf

    $petscplot -t weak --events SNESFunctionEval:SNESJacobianEval:PCApply --stages 1 shaheen/weak_[345]*.out --title '' \
        --legend-labels '$32\times 32\times 3$ coarse level, Redundant' \
        -o figures/shaheen-weak.pdf

    $petscplot -t snes --solve-spec '1:' output/x.80km.m16p2l6.ew.log -o figures/x-80km-m16p2l6-ew.pdf

    $petscplot -t snes --solve-spec '1:' output/y.10km.m10p6l5.ew.log -o figures/y-10km-m10p6l5-ew.pdf

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
    *)
        echo "usage: $0 {figures|pdf}"
esac
