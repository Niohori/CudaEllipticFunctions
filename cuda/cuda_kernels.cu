#include <stdio.h>
#include <stdlib.h>
#include "string"
#include <math_constants.h>
#include <math.h>
#include <random>
#include <iostream>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

//const int nBytes = 256;
//int N = 0;
//int DIM = 0;

__device__ const double PI = 3.14159265359;

//Alternative (Calson's elliptic integral)
// rf(x,y,z) computes Calson's elliptic integral of the first kind, R_F(x,y,z).
// x, y, and z must be nonnegative, and at most one can be zero. See "Numerical
// Recipes in C" by W. H. Press et al. (Chapter 6)
__device__ double rf(double x, double y, double z) {
	const double ERRTOL = 0.0003, rfTINY = 3. * DBL_MIN, rfBIG = DBL_MAX / 3., THIRD = 1.0 / 3.0;
	const double C1 = 1.0 / 24.0, C2 = 0.1, C3 = 3.0 / 44.0, C4 = 1.0 / 14.0;
	double alamb, ave, delx, dely, delz, e2, e3, sqrtx, sqrty, sqrtz, xt, yt, zt;

	if ((fmin(fmin(x, y), z) < 0.0) || (fmin(fmin(x + y, x + z), y + z) < rfTINY) || (fmax(fmax(x, y), z) > rfBIG)) { return 0; }
	xt = x;
	yt = y;
	zt = z;
	do {
		sqrtx = sqrt(xt);
		sqrty = sqrt(yt);
		sqrtz = sqrt(zt);
		alamb = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz;
		xt = 0.25 * (xt + alamb);
		yt = 0.25 * (yt + alamb);
		zt = 0.25 * (zt + alamb);
		ave = THIRD * (xt + yt + zt);
		delx = (ave - xt) / ave;
		dely = (ave - yt) / ave;
		delz = (ave - zt) / ave;
	} while (fmax(fmax(fabs(delx), fabs(dely)), fabs(delz)) > ERRTOL);
	e2 = delx * dely - delz * delz;
	e3 = delx * dely * delz;
	return (1.0 + (C1 * e2 - C2 - C3 * e3) * e2 + C4 * e3) / sqrt(ave);
}

__device__ double CompleteEllipticIntegralFirstKind(double m)
{
	if (m == 1.0) m = 1.0 - 1e-8;
	return rf(0, 1.0 - m, 1.0);
}

__device__  double IncompleteEllipticIntegralFirstKind(double phi, double m)
{
	if (m == 1.0) m = 0.99999999;
	if (phi == 0.0) return 0.0;

	int k = 0;
	while (fabs(phi) > PI / 2.) { (phi > 0) ? k++ : k--; phi += (phi > 0) ? -PI : +PI; }

	double s2 = pow(sin(phi), 2);
	double ell_f = (phi > 0 ? +1 : -1) * sqrt(s2) * rf(1 - s2, 1.0 - s2 * m, 1.0);
	if (k != 0) ell_f += 2. * k * CompleteEllipticIntegralFirstKind(m);
	return ell_f;
}
__device__ void jacobi_sncndn(double u, double m, double* sn, double* cn, double* dn)
{
	if (m == 1.0) m = 0.999999999;

	const double CA = 1.0e-8;
	int bo;
	int i, ii, l;
	double a, b, c, d, emc;
	double em[13], en[13];

	d = 1.0;
	emc = 1.0 - m; // the algorithm is for modulus (1-m)
	if (emc != 0.0) {
		bo = (emc < 0.0);
		if (bo) {
			d = 1.0 - emc;
			emc /= -1.0 / d;
			u *= (d = sqrt(d));
		}
		a = 1.0;
		*dn = 1.0;
		for (i = 0; i < 13; i++) {
			l = i;
			em[i] = a;
			en[i] = (emc = sqrt(emc));
			c = 0.5 * (a + emc);
			if (fabs(a - emc) <= CA * a) break;
			emc *= a;
			a = c;
		}
		u *= c;
		*sn = sin(u);
		*cn = cos(u);
		if (*sn != 0.0) {
			a = (*cn) / (*sn);
			c *= a;
			for (ii = l; ii >= 0; ii--) {
				b = em[ii];
				a *= c;
				c *= *dn;
				*dn = (en[ii] + a) / (b + a);
				a = c / b;
			}
			a = 1.0 / sqrt(c * c + 1.0);
			*sn = ((*sn) >= 0.0 ? a : -a);
			*cn = c * (*sn);
		}
		if (bo) {
			a = *dn;
			*dn = *cn;
			*cn = a;
			*sn /= d;
		}
	}
	else {
		*cn = 1.0 / cosh(u);
		*dn = *cn;
		*sn = tanh(u);
	}
}
__device__ double elliptic_f_sin(double u, double m)
{
	double snx, cnx, dnx;
	jacobi_sncndn(u, m, &snx, &cnx, &dnx);
	return snx;
}

//
/**
===================================================================================================================================
* @brief All elliptic functions and integrals are calculated according the algorithms of Toshio Fukushima:
*@brief "Precise and Fast Computation of Elliptic Integrals and Functions" (DOI: 10.1109/ARITH.2015.15)
*
* @return
=====================================================================================================================================*/
__device__ double sign(double x) {
	return (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
}

// Define the polynomial evaluation functions
__device__ double poly1(double x) {
	return 1.59100345379079220f + x * (0.41600074399178694f + x * (0.24579151426410342f + x * (0.17948148291490615f + x * (0.14455605708755515f + x * (0.123200993312427720f + x * (0.108938811574293530f + x * (0.098853409871592910f + x * (0.091439629201749750f + x * (0.08584259159541390f + x * (0.08154111871830322f))))))))));
}

__device__ double poly2(double x) {
	return 1.63525673226458000 + (x * (0.47119062614873230 + (x * (0.30972841083149960 + (x * (0.25220831177313570 + (x * (0.22672562321968465 + (x * (0.215774446729585980 + (x * (0.213108771877348920 + (x * (0.216029124605188280 + (x * (0.223255831633057900 + (x * (0.23418050129420992 + (x * (0.24855768297226408 + (x * (0.266363809892617540))))))))))))))))))))));
}

__device__ double poly3(double x) {
	return 1.68575035481259600 + (x * (0.54173184861328030 + (x * (0.40152443839069024 + (x * (0.36964247342088910 + (x * (0.37606071535458363 + (x * (0.405235887085125900 + (x * (0.453294381753999050 + (x * (0.520518947651184200 + (x * (0.609426039204995000 + (x * (0.72426352228290890 + (x * (0.87101384770981240 + (x * (1.057652872753547000))))))))))))))))))))));
}

__device__ double poly4(double x) {
	return 1.74435059722561330 + (x * (0.63486427537193530 + (x * (0.53984256416444550 + (x * (0.57189270519378740 + (x * (0.67029513626540620 + (x * (0.832586590010977200 + (x * (1.073857448247933300 + (x * (1.422091460675497700 + (x * (1.920387183402304700 + (x * (2.63255254833165430 + (x * (3.65210974731903940 + (x * (5.115867135558866000 + (x * (7.224080007363877000))))))))))))))))))))))));
}

__device__ double poly5(double x) {
	return 1.81388393681698260 + (x * (0.76316324570055730 + (x * (0.76192860532159580 + (x * (0.95107465366842790 + (x * (1.31518067170316100 + (x * (1.928560693477410900 + (x * (2.937509342531378700 + (x * (4.594894405442878000 + (x * (7.330071221881720000 + (x * (11.8715125974253010 + (x * (19.4585137482293800 + (x * (32.20638657246427000 + (x * (53.73749198700555000 + (x * (90.27388602941000000))))))))))))))))))))))))));
}

__device__ double poly6(double x) {
	return 1.89892491027155350 + (x * (0.95052179461824450 + (x * (1.15107758995901580 + (x * (1.75023910698630060 + (x * (2.95267681263687500 + (x * (5.285800396121451000 + (x * (9.832485716659980000 + (x * (18.78714868327559600 + (x * (36.61468615273698000 + (x * (72.4529239512777100 + (x * (145.107957734706900 + (x * (293.4786396308497000 + (x * (598.3851815055010000 + (x * (1228.420013075863400 + (x * (2536.529755382764500))))))))))))))))))))))))))));
}

__device__ double poly7(double x) {
	return 2.00759839842437640 + (x * (1.24845723121234740 + (x * (1.92623465707647970 + (x * (3.75128964008758770 + (x * (8.11994455493204500 + (x * (18.66572130873555200 + (x * (44.60392484291437400 + (x * (109.5092054309498300 + (x * (274.2779548232414000 + (x * (697.559800860632700 + (x * (1795.71601450024720 + (x * (4668.381716790390000 + (x * (12235.76246813664300 + (x * (32290.17809718321000 + (x * (85713.07608195965000 + (x * (228672.1890493117 + (x * (612757.27119158520))))))))))))))))))))))))))))))));
}

__device__ double poly8(double x) {
	return 2.15651564749964340 + (x * (1.79180564184946320 + (x * (3.82675128746571320 + (x * (10.3867246836379720 + (x * (31.4033140546807030 + (x * (100.9237039498695500 + (x * (337.3268282632273000 + (x * (1158.707930567827800 + (x * (4060.990742193632300 + (x * (14454.0018403434480 + (x * (52076.6610759940450 + (x * (189493.6591462156800 + (x * (695184.5762413896000 + (x * (2567994.048255285000 + (x * (9541921.966748387000 + (x * (35634927.44218076 + (x * (133669298.46120408 + (x * (503352186.68662846 + (x * (1901975729.538660 + (x * 7208915015.33010400)))))))))))))))))))))))))))))))))))));
}

__device__ double poly9(double x) {
	return 2.31812262171251060 + (x * (2.61692015029123270 + (x * (7.89793507573135600 + (x * (30.5023971544667240 + (x * (131.486936552352860 + (x * (602.9847637356492000 + (x * (2877.024617809973000 + (x * (14110.51991915180400 + (x * (70621.44088156540000 + (x * (358977.266582531000 + (x * (1847238.26372397180 + (x * (9600515.416049214000 + (x * (50307677.08502367000 + (x * (265444188.6527128000 + (x * (1408862325.028702700 + (x * (7515687935.373775))))))))))))))))))))))))))))));
}

__device__ double poly10(double x) {
	return 2.47359617375134400 + (x * (3.72762424411809900 + (x * (15.6073930355493060 + (x * (84.1285084280588800 + (x * (506.981819704061370 + (x * (3252.277058145123600 + (x * (21713.24241957434400 + (x * (149037.0451890932700 + (x * (1043999.331089990800 + (x * (7427974.81704203900 + (x * (53503839.6755866100 + (x * (389249886.9948708400 + (x * (2855288351.100810500 + (x * (21090077038.76684000 + (x * (156699833947.7902000 + (x * (1170222242422.440 + (x * (8777948323668.9375 + (x * (66101242752484.950 + (x * (499488053713388.8 + (x * 37859743397240296.0)))))))))))))))))))))))))))))))))))));
}

__device__ double poly11(double x) {
	return (x * (0.06250000000000000 + (x * (0.03125000000000000 + (x * (0.02050781250000000 + (x * (0.01513671875000000 + (x * (0.011934280395507812 + (x * (0.009816169738769531 + (x * (0.008315593004226685 + (x * (0.007199153304100037 + (x * (0.00633745662344154 + (x * (0.00565311038371874 + (x * (0.005097046040418718 + (x * (0.004636680381850056 + (x * (0.004249547423822886 + (x * (0.003919665602267974))))))))))))))))))))))))))));
}
__device__ double poly12(double x) {
	return 1.59100345379079220 + (x * (0.41600074399178694 + (x * (0.24579151426410342 + (x * (0.17948148291490615 + (x * (0.14455605708755515 + (x * (0.123200993312427720 + (x * (0.108938811574293530 + (x * (0.098853409871592910 + (x * (0.091439629201749750 + (x * (0.08584259159541390 + (x * (0.08154111871830322))))))))))))))))))));
}

__device__ double serf(double y, double m) {
	return 1.0 + y * (0.166667 + 0.166667 * m +
		y * (0.075 + (0.05 + 0.075 * m) * m +
			y * (0.0446429 + m * (0.0267857 + (0.0267857 + 0.0446429 * m) * m) +
				y * (0.0303819 +
					m * (0.0173611 +
						m * (0.015625 + (0.0173611 + 0.0303819 * m) * m)) +
					y * (0.0223722 +
						m * (0.012429 +
							m * (0.0106534 +
								m * (0.0106534 + (0.012429 + 0.0223722 * m) * m))) +
						y * (0.0173528 +
							m * (0.00946514 +
								m * (0.00788762 +
									m * (0.00751202 +
										m * (0.00788762 + (0.00946514 + 0.0173528 * m) * m)))) +
							y * (0.01396480 +
								m * (0.00751953 +
									m * (0.00615234 +
										m * (0.00569661 +
											m * (0.00569661 +
												m * (0.00615234 + (0.00751953 +
													0.0139648 * m) * m))))) +
								y * (0.01155180 +
									m * (0.00616096 +
										m * (0.00497616 +
											m * (0.00452378 +
												m * (0.00439812 + m * (0.00452378 +
													m * (0.00497616 + (0.00616096 +
														0.0115518 * m) * m)))))) + (0.00976161 +
															m * (0.00516791 +
																m * (0.00413433 +
																	m * (0.00371030 + m * (0.00354165 +
																		m * (0.00354165 + m * (0.00371030 +
																			m * (0.00413433 + (0.00516791 +
																				0.00976161 * m) * m)))))))) * y))))))));
}
__device__ double asn(double s, double m) {
	double yA = 0.04094 - 0.00652 * m;
	double y = s * s;

	if (y < yA) {
		return s * serf(y, m);
	}

	double p = 1.0;
	for (int i = 0; i < 10; ++i) {
		y /= ((1.0f + sqrt(1.0 - y)) * (1.0f + sqrt(1.0 - m * y)));
		p += p;
		if (y < yA) {
			return p * sqrt(y) * serf(y, m);
		}
	}

	return NAN;
}

__device__ double acn(double c, double m) {
	double mc = 1.0 - m;
	double x = c * c;
	double p = 1.0;

	for (int i = 0; i < 10; ++i) {
		if (x > 0.5) {
			return p * asn(sqrt(1.0 - x), m);
		}

		double d = sqrt(mc + m * x);
		x = (sqrt(x) + d) / (1.0 + d);
		p += p;
	}

	return NAN;
}

// Define the CUDA kernel for evaluating complete elleiptic integral of the first kind K
__device__ double K(double m) {
	if (m < 1.0f) {
		double t;
		double x = m;// *m;
		if (x < 0.1) {
			t = poly1(x - 0.05);
		}
		else if (x < 0.2)
			t = poly2(x - 0.15);
		else if (x < 0.3)
			t = poly3(x - 0.25);
		else if (x < 0.4)
			t = poly4(x - 0.35);
		else if (x < 0.5)
			t = poly5(x - 0.45);
		else if (x < 0.6)
			t = poly6(x - 0.55);
		else if (x < 0.7)
			t = poly7(x - 0.65);
		else if (x < 0.8)
			t = poly8(x - 0.75);
		else if (x < 0.85)
			t = poly9(x - 0.825);
		else if (x < 0.9)
			t = poly10(x - 0.875);
		else {
			// Handle the last condition
			double td = 1.0 - x;
			double qd = poly11(td);
			double kdm = poly12(td - 0.05);
			t = -log(qd) * (kdm / 3.14159265358979323846);
		}
		return t;
	}
	else {
		return (double)INFINITY;
	}
}

__device__ double _rawF(double sin_phi, double m) {
	double yS = 0.9000308778823196;
	if (m == 0.0) {
		return asin(sin_phi);
	}
	if (m == 1.0) {
		return atanh(sin_phi);
	}

	double sin_phi2 = sin_phi * sin_phi;
	if (sin_phi2 <= yS) {
		return asn(sin_phi, m);
	}

	double mc = 1.0 - m;
	double c = sqrt(1.0 - sin_phi2);
	double x = c * c;
	double d2 = mc + m * x;
	double z = c / sqrt(mc + m * c * c);

	if (x < yS * d2) {
		return K(m) - asn(c / sqrt(d2), m);
	}

	double v = mc * (1.0 - x);
	if (v < x * d2) {
		return acn(c, m);
	}

	return K(m) - acn(sqrt(v / d2), m);
}

__device__ double _F(double phi, double m) {
	if (fabs(phi) < PI / 2) {
		return sign(phi) * _rawF(sin(phi), m);
	}

	double j = floor(phi / PI);
	double new_phi = phi - j * PI;
	double sign_phi = sign(new_phi);

	if (fabs(new_phi) > PI / 2) {
		j += sign_phi;
		new_phi -= sign_phi * PI;
	}

	sign_phi = sign(new_phi);
	return 2 * j * K(m) + sign_phi * _rawF(sinf(fabs(new_phi)), m);
}
// Define the CUDA kernel for evaluating incomplete elleiptic integral of the first kind F
__device__ double F(double phi, double m) {
	if (m > 1.0) {
		double m12 = sqrt(m);
		double theta = asin(m12 * sin(phi));
		double sign_theta = sign(theta);
		double abs_theta = abs(theta);
		return sign_theta / m12 * _F(abs_theta, 1.0 / m);
	}
	else if (m < 0.0) {
		double n = -m;
		double m12 = 1.0f / sqrt(1.0 + n);
		double m1m = n / (1.0 + n);
		double new_phi = PI / 2 - phi;
		double sign_phi = sign(new_phi);
		double abs_phi = abs(new_phi);
		return m12 * K(m1m) - sign_phi * m12 * _F(abs_phi, m1m);
	}
	double abs_phi = abs(phi);
	double sign_phi = sign(phi);
	return sign_phi * _F(abs_phi, m);
}

__device__ void _DeltaXNloop(double u, double m, double n, double& s1, double& s2, double& s3) {
	double sn, cn, dn;
	double up = u / (pow(2.0, n));
	double up2 = up * up;

	sn = up * (up2 * (up2 * ((m * ((-(m / 5040.0) - 3.0 / 112.0) * m - 3.0 / 112.0) - 1.0 / 5040.0) * up2 + (m / 120 + 7.0 / 60.0) * m + 1.0 / 120) - m / 6 - 1.0 / 6) + 1);

	cn = 1 + up2 * (-(1.0 / 2.0) + up2 * (1.0 / 24 + m / 6 + up2 * (-(1.0 / 720) + (-(11.0 / 180.0) - m / 45) * m + (-(1.0 / 40320) + m * (-(17.0 / 1680) + (-(19.0 / 840) - m / 630) * m)) * up2)));

	dn = m * (m * (pow(up2, 2) * (1.0 / 24.0 - (11.0 * up2) / 180.0) - (m * pow(up2, 3)) / 720.0) + (up2 * (1.0 / 6 - up2 / 45) - 0.5) * up2) + 1;

	double Deltasn = up - sn;
	double Deltacn = 1 - cn;
	double Deltadn = 1 - dn;

	for (int i = 0; i < n; i++) {
		double sn2 = sn * sn;
		double sn4 = sn2 * sn2;

		double den = 1.0 / (1 - m * sn4);
		Deltasn = 2 * (Deltasn * cn * dn + up * (Deltacn * (1 - Deltadn) + Deltadn - m * sn4)) * den;
		Deltacn = ((1 + cn) * Deltacn + sn2 - 2 * m * sn4) * den;
		Deltadn = ((1 + dn) * Deltadn + m * (sn2 - 2 * sn4)) * den;

		up += up;
		sn = up - Deltasn;
		cn = 1.0 - Deltacn;
		dn = 1.0 - Deltadn;
	}
	s1 = sn;
	s2 = cn;
	s3 = dn;

	return;
}

__device__ void fold_0_25(double u1, double m, double kp, double& s1, double& s2, double& s3) {
	double ss1, ss2, ss3;
	if (u1 == 0) {
		s1 = 0.0;
		s2 = 1.0;
		s3 = 1.0;
		return;
	}

	_DeltaXNloop(u1, m, (u1 > 0.0) ? fmax(6.0 + (floor(log2(u1))), 1.0) : 0.0, ss1, ss2, ss3);
	double den = 1.0 / (1 + kp - m * ss1 * ss1);

	s1 = den * sqrt(1.0 + kp) * (ss2 * ss3 - kp * ss1);
	s2 = den * sqrt(kp * (1.0 + kp)) * (ss2 + ss1 * ss3);
	s3 = den * sqrt(kp) * ((1.0 + kp) * ss3 + m * ss1 * ss2);
	return;
}

__device__ void fold_0_50(double u1, double m, double Kscreen, double Kactual, double kp, double& s1, double& s2, double& s3) {
	double ss1, ss2, ss3;
	if (u1 == 0) {
		s1 = 0.0;
		s2 = 1.0;
		s3 = 1.0;
		return;
	}

	if (u1 > 0.25 * Kscreen) {
		fold_0_25(Kactual / 2.0 - u1, m, kp, ss1, ss2, ss3);
		s1 = ss1;
		s2 = ss2;
		s3 = ss3;
	}
	else {
		_DeltaXNloop(u1, m, (u1 > 0.0) ? fmax(6.0 + (floor(log2(u1))), 1.0) : 0.0, ss1, ss2, ss3);
		s1 = ss1;
		s2 = ss2;
		s3 = ss3;
	}
	double sn = s1;
	double cn = s2;
	double dn = s3;
	s1 = cn / dn;
	s2 = kp * sn / dn;
	s3 = kp / dn;

	return;
}

__device__ void fold_1_00(double u1, double m, double Kscreen, double Kactual, double kp, double& s1, double& s2, double& s3) {
	double ss1, ss2, ss3;
	if (u1 == 0.0) {
		s1 = 0.0;
		s2 = 1.0;
		s3 = 1.0;
		return;
	}

	if (u1 > 0.5 * Kscreen) {
		fold_0_50(Kactual - u1, m, Kscreen, Kactual, kp, ss1, ss2, ss3);
		s1 = ss1;
		s2 = ss2;
		s3 = ss3;
	}
	else if (u1 > 0.25 * Kscreen) {
		fold_0_25(Kactual * 0.5 - u1, m, kp, ss1, ss2, ss3);
		s1 = ss1;
		s2 = ss2;
		s3 = ss3;
	}
	else {
		_DeltaXNloop(u1, m, (u1 > 0) ? fmax(6.0 + (floor(log2(u1))), 1.0) : 0.0, ss1, ss2, ss3);
		s1 = ss1;
		s2 = ss2;
		s3 = ss3;
	}
	double sn = s1;
	double cn = s2;
	double dn = s3;
	s1 = cn / dn;
	s2 = -kp * sn / dn;
	s3 = kp / dn;
	return;
}

__device__ double  _rawSN(double u, double m, double Kscreen, double  Kactual, double kp) {
	double ss1, ss2, ss3;
	bool check = (u >= 2.0 * Kscreen);
	double sign = check ? -1.0 : 1.0;
	u = check ? u - 2.0 * Kactual : u;
	if (u > Kscreen)
	{
		fold_1_00(u - Kactual, m, Kscreen, Kactual, kp, ss1, ss2, ss3);

		return  sign * ss2;
	}
	if (u == Kscreen) return 1.0;
	if (u > 0.5 * Kscreen) {
		fold_0_50(Kactual - u, m, Kscreen, Kactual, kp, ss1, ss2, ss3);
		return sign * ss2;
	}
	if (u == Kscreen * 0.5) return 1.0 / sqrt(1.0 + kp);
	if (u >= 0.25 * Kscreen) {
		fold_0_25(Kactual / 2.0 - u, m, kp, ss1, ss2, ss3);
		return sign * ss2;
	}
	_DeltaXNloop(u, m, u > 0.0 ? fmax(6.0 + (floor(log2(u))), 1.0) : 0.0, ss1, ss2, ss3);
	return  sign * ss2;
}

__device__ double _SN(double u, double m) {
	double tempK = K(m);
	double result;

	if (u > 4.0 * tempK) {
		result = _rawSN(fmod(u, 4.0 * tempK), m, tempK, tempK, sqrt(1.0 - m));
	}
	else {
		result = _rawSN(u, m, tempK, tempK, sqrt(1.0 - m));
	}
	return result;
}

__device__ double sn(double u, double m) {
	double signu = sign(u);
	u = abs(u);
	if (m < 1.0) {
		//return signu * _SN(u, m);
		if (abs(_SN(u, m)) > 0.99999999999999) { return signu * _SN(u, m); }
		return signu * sqrt(1.0 - pow(signu * _SN(u, m), 2));
	}
	double sqrtm = sqrt(m);
	return signu * sqrt(1.0 - pow(signu * _SN(u * sqrtm, 1.0 / m) / sqrtm, 2));
}

__device__ double sn2(double u, double m) {
	const double tolerance = 1.0e-8; // Tolerance for convergence
	const int max_iterations = 100; // Maximum number of iterations

	double sn_u = u;
	double delta_sn_u = 0.0;
	double sn_old = 0.0;

	for (int i = 0; i < max_iterations; ++i) {
		sn_old = sn_u;
		sn_u = sin(u + m * sn_u);
		delta_sn_u = sn_u - sn_old;

		// If the change in sn_u is small enough, we assume convergence
		if (fabs(delta_sn_u) < tolerance)
			break;
	}

	return sn_u;
}

__global__ void computeEllipticIntegralK(double m, double* F_result) {
	*F_result = K(m);
}
__global__ void computeEllipticIntegralFukushima(double theta, double m, double* F_result) {
	*F_result = F(theta, m);
}

__global__ void computeIncompleteEllipticIntegralSimple(double theta, double m, double* F_result) {
	*F_result = IncompleteEllipticIntegralFirstKind(theta, m);
}
__global__ void computeCompleteEllipticIntegralSimple(double m, double* F_result) {
	*F_result = CompleteEllipticIntegralFirstKind(m);
}

__global__ void computeJacobiSN(double arg, double m, double* F_result) {
	*F_result = sn(arg, m);
}

__global__ void computeJacobiSnSimple(double arg, double m, double* F_result) {
	*F_result = elliptic_f_sin(arg, m);
}

void cudaInCompleteEllipticIntegralFukushima(double theta, double k, double* result) {
	double m = k * k;
	computeEllipticIntegralFukushima << <1, 1 >> > (theta, m, result);
}

void cudaInCompleteEllipticIntegralSimple(double theta, double k, double* result) {
	double m = k * k;
	computeIncompleteEllipticIntegralSimple << <1, 1 >> > (theta, m, result);
}

void cudaCompleteEllipticIntegralSimple(double k, double* result) {
	double m = k * k;
	computeCompleteEllipticIntegralSimple << <1, 1 >> > (m, result);
}

void cudaCompleteEllipticIntegralK(double k, double* result) {
	double m = k * k;
	computeEllipticIntegralK << <1, 1 >> > (m, result);
}

void cudaJacobiSN(double k, double arg, double* result) {
	double m = k * k;
	computeJacobiSN << <1, 1 >> > (arg, m, result);
	//jacobi_sn_kernel << <1, 1 >> > (arg, k, result);
}

void cudaJacobiSnSimple(double k, double phi, double* result) {
	double m = k * k;
	computeJacobiSnSimple << <1, 1 >> > (phi, m, result);
	//jacobi_sn_kernel << <1, 1 >> > (arg, k, result);
}
