static float rope_frequencies[2][32];
static pthread_once_t rope_once=PTHREAD_ONCE_INIT;
/* Match pinned ds4's powf/reciprocal and YaRN operation order. */
#pragma float_control(precise, on, push)
static void init_rope(void) {
    const double pi=3.14159265358979323846;
    for(int kind=0;kind<2;kind++) {
        float base=kind?160000.0f:10000.0f;
        float low=(float)floor(64.0*log(65536.0/(32.0*2.0*pi))/(2.0*log(base)));
        float high=(float)ceil(64.0*log(65536.0/(2.0*pi))/(2.0*log(base)));
        for(int i=0;i<32;i++) {
            float denominator=powf(base,(float)i/32.0f),f=1.0f/denominator;
            if(kind) {
                float ramp=fminf(1.0f,fmaxf(0.0f,(i-low)/(high-low)));
                float smooth=1.0f-ramp;
                float interpolate=(f/16.0f)*(1.0f-smooth),extrapolate=f*smooth;
                f=interpolate+extrapolate;
            }
            rope_frequencies[kind][i]=f;
        }
    }
}
#pragma float_control(pop)
