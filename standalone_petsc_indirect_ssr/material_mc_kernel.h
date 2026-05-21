/*
 * Per-integration-point 3D Mohr-Coulomb constitutive kernel lifted from the
 * Octave implementation in https://github.com/Beremi/slope_stability_octave.
 *
 * Inputs use engineering shear strains. DS is returned as a column-major 6x6
 * block (36 entries) matching the MATLAB / Octave storage convention.
 */

#ifndef SLOPE_STABILITY_PETSC_CONSTITUTIVE_3D_KERNEL_H
#define SLOPE_STABILITY_PETSC_CONSTITUTIVE_3D_KERNEL_H

#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline void eigproj(const double *esq, const double *et,
                           double eig_other_sum, double eig_other_prod,
                           double inv_denom, double *P)
{
    int i;
    for (i = 0; i < 3; i++)
        P[i] = inv_denom * (esq[i] - eig_other_sum * et[i] + eig_other_prod);
    for (i = 3; i < 6; i++)
        P[i] = inv_denom * (esq[i] - eig_other_sum * et[i]);
}

static inline void build_der_esq(const double *et, double *d)
{
    memset(d, 0, 36 * sizeof(double));
    d[0] = 2.0 * et[0];
    d[3] = et[3];
    d[5] = et[5];
    d[7] = 2.0 * et[1];
    d[9] = et[3];
    d[10] = et[4];
    d[14] = 2.0 * et[2];
    d[16] = et[4];
    d[17] = et[5];
    d[18] = et[3];
    d[19] = et[3];
    d[21] = 0.5 * (et[0] + et[1]);
    d[22] = 0.5 * et[5];
    d[23] = 0.5 * et[4];
    d[25] = et[4];
    d[26] = et[4];
    d[27] = 0.5 * et[5];
    d[28] = 0.5 * (et[1] + et[2]);
    d[29] = 0.5 * et[3];
    d[30] = et[5];
    d[32] = et[5];
    d[33] = 0.5 * et[4];
    d[34] = 0.5 * et[3];
    d[35] = 0.5 * (et[0] + et[2]);
}

static inline double ident_flat(int i, int j)
{
    if (i != j)
        return 0.0;
    return (i < 3) ? 1.0 : 0.5;
}

static inline double iota(int i) { return (i < 3) ? 1.0 : 0.0; }

static inline void constitutive_3D_point(
    const double *E_in,
    double c_bar, double sin_phi, double shear, double bulk, double lame,
    double *S, double *DS)
{
    double sin_psi = sin_phi;

    double et[6] = {E_in[0], E_in[1], E_in[2],
                    0.5 * E_in[3], 0.5 * E_in[4], 0.5 * E_in[5]};

    double esq[6];
    esq[0] = et[0] * et[0] + et[3] * et[3] + et[5] * et[5];
    esq[1] = et[1] * et[1] + et[3] * et[3] + et[4] * et[4];
    esq[2] = et[2] * et[2] + et[4] * et[4] + et[5] * et[5];
    esq[3] = et[0] * et[3] + et[1] * et[3] + et[4] * et[5];
    esq[4] = et[3] * et[5] + et[1] * et[4] + et[2] * et[4];
    esq[5] = et[0] * et[5] + et[3] * et[4] + et[2] * et[5];

    double I1 = et[0] + et[1] + et[2];
    double I2 = et[0] * et[1] + et[0] * et[2] + et[1] * et[2] - et[3] * et[3] - et[4] * et[4] - et[5] * et[5];
    double I3 = et[0] * et[1] * et[2] - et[2] * et[3] * et[3] - et[1] * et[5] * et[5] - et[0] * et[4] * et[4] + 2.0 * et[3] * et[4] * et[5];

    double Q = (I1 * I1 - 3.0 * I2) / 9.0;
    if (Q < 0.0)
        Q = 0.0;
    double R = (-2.0 * I1 * I1 * I1 + 9.0 * I1 * I2 - 27.0 * I3) / 54.0;
    double sqQ = sqrt(Q);
    double theta;
    if (Q < 1e-30)
    {
        theta = 0.0;
    }
    else
    {
        double ca = R / (sqQ * Q);
        if (ca < -1.0)
            ca = -1.0;
        if (ca > 1.0)
            ca = 1.0;
        theta = acos(ca) / 3.0;
    }

    double e1 = -2.0 * sqQ * cos(theta + 2.0 * M_PI / 3.0) + I1 / 3.0;
    double e2 = -2.0 * sqQ * cos(theta - 2.0 * M_PI / 3.0) + I1 / 3.0;
    double e3 = -2.0 * sqQ * cos(theta) + I1 / 3.0;

    double f_tr = 2.0 * shear * ((1.0 + sin_phi) * e1 - (1.0 - sin_phi) * e3) + 2.0 * lame * sin_phi * I1 - c_bar;

    if (f_tr <= 0.0)
    {
        double lI1 = lame * I1, s2 = 2.0 * shear;
        S[0] = lI1 + s2 * et[0];
        S[1] = lI1 + s2 * et[1];
        S[2] = lI1 + s2 * et[2];
        S[3] = s2 * et[3];
        S[4] = s2 * et[4];
        S[5] = s2 * et[5];
        if (DS)
        {
            memset(DS, 0, 36 * sizeof(double));
            DS[0] = s2 + lame;
            DS[7] = s2 + lame;
            DS[14] = s2 + lame;
            DS[21] = shear;
            DS[28] = shear;
            DS[35] = shear;
            DS[1] = lame;
            DS[6] = lame;
            DS[2] = lame;
            DS[12] = lame;
            DS[8] = lame;
            DS[13] = lame;
        }
        return;
    }

    double gsl = (e1 - e2) / (1.0 + sin_psi);
    double gsr = (e2 - e3) / (1.0 - sin_psi);
    double den_s = 4.0 * lame * sin_phi * sin_psi + 4.0 * shear * (1.0 + sin_phi * sin_psi);
    double lam_s = f_tr / den_s;

    if (lam_s <= gsl && lam_s <= gsr)
    {
        double id1 = 1.0 / ((e1 - e2) * (e1 - e3));
        double id2 = 1.0 / ((e2 - e1) * (e2 - e3));
        double id3 = 1.0 / ((e3 - e1) * (e3 - e2));

        double P1[6], P2[6], P3[6];
        eigproj(esq, et, e2 + e3, e2 * e3, id1, P1);
        eigproj(esq, et, e1 + e3, e1 * e3, id2, P2);
        eigproj(esq, et, e1 + e2, e1 * e2, id3, P3);

        double sig1 = lame * I1 + 2.0 * shear * e1 - lam_s * (2.0 * lame * sin_psi + 2.0 * shear * (1.0 + sin_psi));
        double sig2 = lame * I1 + 2.0 * shear * e2 - lam_s * 2.0 * lame * sin_psi;
        double sig3 = lame * I1 + 2.0 * shear * e3 - lam_s * (2.0 * lame * sin_psi - 2.0 * shear * (1.0 - sin_psi));

        {
            int i;
            for (i = 0; i < 6; i++)
                S[i] = sig1 * P1[i] + sig2 * P2[i] + sig3 * P3[i];
        }

        if (DS)
        {
            double der[36];
            build_der_esq(et, der);

            double Dp[6], Dq[6];
            {
                int i;
                for (i = 0; i < 6; i++)
                {
                    double iot = iota(i);
                    Dp[i] = 2.0 * shear * ((1.0 + sin_phi) * P1[i] - (1.0 - sin_phi) * P3[i]) + 2.0 * lame * sin_phi * iot;
                    Dq[i] = 2.0 * shear * ((1.0 + sin_psi) * P1[i] - (1.0 - sin_psi) * P3[i]) + 2.0 * lame * sin_psi * iot;
                }
            }

            double em23 = e2 + e3, em13 = e1 + e3, em12 = e1 + e2;
            double c1a = 2.0 * e1 - e2 - e3, c1b = e2 - e3;
            double c2a = 2.0 * e2 - e1 - e3, c2b = e1 - e3;
            double c3a = 2.0 * e3 - e1 - e2, c3b = e1 - e2;
            double inv_den = 1.0 / den_s;
            int j, i;
            for (j = 0; j < 6; j++)
            {
                for (i = 0; i < 6; i++)
                {
                    int k = i + 6 * j;
                    double id_k = ident_flat(i, j);
                    double o11 = P1[i] * P1[j];
                    double o22 = P2[i] * P2[j];
                    double o33 = P3[i] * P3[j];
                    double E1d = id1 * (der[k] - id_k * em23 - c1a * o11 - c1b * (o22 - o33));
                    double E2d = id2 * (der[k] - id_k * em13 - c2a * o22 - c2b * (o11 - o33));
                    double E3d = id3 * (der[k] - id_k * em12 - c3a * o33 - c3b * (o11 - o22));
                    DS[k] = sig1 * E1d + sig2 * E2d + sig3 * E3d + lame * iota(i) * iota(j) + 2.0 * shear * (o11 + o22 + o33) - Dq[i] * Dp[j] * inv_den;
                }
            }
        }
        return;
    }

    if (gsl < gsr)
    {
        double den_l = 4.0 * lame * sin_phi * sin_psi + shear * (1.0 + sin_phi) * (1.0 + sin_psi) + 2.0 * shear * (1.0 - sin_phi) * (1.0 - sin_psi);
        double lam_l = (shear * ((1.0 + sin_phi) * (e1 + e2) - 2.0 * (1.0 - sin_phi) * e3) + 2.0 * lame * sin_phi * I1 - c_bar) / den_l;
        double gla = (e1 + e2 - 2.0 * e3) / (3.0 - sin_psi);

        if (lam_l >= gsl && lam_l <= gla)
        {
            double idl3 = 1.0 / ((e3 - e1) * (e3 - e2));
            double P3[6], P12[6];
            eigproj(esq, et, e1 + e2, e1 * e2, idl3, P3);
            {
                int i;
                for (i = 0; i < 6; i++)
                    P12[i] = iota(i) - P3[i];
            }

            double sig1l = lame * I1 + shear * (e1 + e2) - lam_l * (2.0 * lame * sin_psi + shear * (1.0 + sin_psi));
            double sig3l = lame * I1 + 2.0 * shear * e3 - lam_l * (2.0 * lame * sin_psi - 2.0 * shear * (1.0 - sin_psi));

            {
                int i;
                for (i = 0; i < 6; i++)
                    S[i] = sig1l * P12[i] + sig3l * P3[i];
            }

            if (DS)
            {
                double der[36];
                build_der_esq(et, der);
                double em12 = e1 + e2;
                double Dp[6], Dq[6];
                {
                    int i;
                    for (i = 0; i < 6; i++)
                    {
                        double iot = iota(i);
                        Dp[i] = shear * ((1.0 + sin_phi) * P12[i] - 2.0 * (1.0 - sin_phi) * P3[i]) + 2.0 * lame * sin_phi * iot;
                        Dq[i] = shear * ((1.0 + sin_psi) * P12[i] - 2.0 * (1.0 - sin_psi) * P3[i]) + 2.0 * lame * sin_psi * iot;
                    }
                }
                double inv_den = 1.0 / den_l;
                int j, i;
                for (j = 0; j < 6; j++)
                {
                    for (i = 0; i < 6; i++)
                    {
                        int k = i + 6 * j;
                        double id_k = ident_flat(i, j);
                        double o33 = P3[i] * P3[j];
                        double o1212 = P12[i] * P12[j];
                        double o12e = P12[i] * et[j];
                        double oe12 = et[i] * P12[j];
                        double o12_3 = P12[i] * P3[j];
                        double o3_12 = P3[i] * P12[j];
                        double E3d = idl3 * (der[k] - id_k * em12 - (o12e + oe12) + em12 * o1212 + (em12 - 2.0 * e3) * o33 + e3 * (o12_3 + o3_12));
                        DS[k] = (sig3l - sig1l) * E3d + lame * iota(i) * iota(j) + shear * (o1212 + 2.0 * o33) - Dq[i] * Dp[j] * inv_den;
                    }
                }
            }
            return;
        }
    }

    if (gsl > gsr)
    {
        double den_r = 4.0 * lame * sin_phi * sin_psi + 2.0 * shear * (1.0 + sin_phi) * (1.0 + sin_psi) + shear * (1.0 - sin_phi) * (1.0 - sin_psi);
        double lam_r = (shear * (2.0 * (1.0 + sin_phi) * e1 - (1.0 - sin_phi) * (e2 + e3)) + 2.0 * lame * sin_phi * I1 - c_bar) / den_r;
        double gra = (2.0 * e1 - e2 - e3) / (3.0 + sin_psi);

        if (lam_r >= gsr && lam_r <= gra)
        {
            double idr1 = 1.0 / ((e1 - e2) * (e1 - e3));

            double P1[6], P23[6];
            eigproj(esq, et, e2 + e3, e2 * e3, idr1, P1);
            {
                int i;
                for (i = 0; i < 6; i++)
                    P23[i] = iota(i) - P1[i];
            }

            double sig1r = lame * I1 + 2.0 * shear * e1 - lam_r * (2.0 * lame * sin_psi + 2.0 * shear * (1.0 + sin_psi));
            double sig3r = lame * I1 + shear * (e2 + e3) - lam_r * (2.0 * lame * sin_psi - shear * (1.0 - sin_psi));

            {
                int i;
                for (i = 0; i < 6; i++)
                    S[i] = sig1r * P1[i] + sig3r * P23[i];
            }

            if (DS)
            {
                double der[36];
                build_der_esq(et, der);
                double em23 = e2 + e3;
                double Dp[6], Dq[6];
                {
                    int i;
                    for (i = 0; i < 6; i++)
                    {
                        double iot = iota(i);
                        Dp[i] = shear * (2.0 * (1.0 + sin_phi) * P1[i] - (1.0 - sin_phi) * P23[i]) + 2.0 * lame * sin_phi * iot;
                        Dq[i] = shear * (2.0 * (1.0 + sin_psi) * P1[i] - (1.0 - sin_psi) * P23[i]) + 2.0 * lame * sin_psi * iot;
                    }
                }
                double inv_den = 1.0 / den_r;
                int j, i;
                for (j = 0; j < 6; j++)
                {
                    for (i = 0; i < 6; i++)
                    {
                        int k = i + 6 * j;
                        double id_k = ident_flat(i, j);
                        double o11 = P1[i] * P1[j];
                        double o2323 = P23[i] * P23[j];
                        double o23e = P23[i] * et[j];
                        double oe23 = et[i] * P23[j];
                        double o23_1 = P23[i] * P1[j];
                        double o1_23 = P1[i] * P23[j];
                        double E1d = idr1 * (der[k] - id_k * em23 - (o23e + oe23) + em23 * o2323 + (em23 - 2.0 * e1) * o11 + e1 * (o23_1 + o1_23));
                        DS[k] = (sig1r - sig3r) * E1d + lame * iota(i) * iota(j) + shear * (2.0 * o11 + o2323) - Dq[i] * Dp[j] * inv_den;
                    }
                }
            }
            return;
        }
    }

    {
        double sig_a = c_bar / (2.0 * sin_phi);
        S[0] = sig_a;
        S[1] = sig_a;
        S[2] = sig_a;
        S[3] = 0.0;
        S[4] = 0.0;
        S[5] = 0.0;
        if (DS)
            memset(DS, 0, 36 * sizeof(double));
    }
}

#endif
