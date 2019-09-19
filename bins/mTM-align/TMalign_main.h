/*
===============================================================================
Implementation of TM-align in C/C++

This program is written by Jianyi Yang at
Yang Zhang lab
Center for Computational Medicine and Bioinformatics
University of Michigan
100 Washtenaw Avenue, Ann Arbor, MI 48109-2218


Please report bugs and questions to yangji@umich.edu or zhng@umich.edu
===============================================================================
*/
#define MAXLEN 10000                        //maximum length of filenames
char version[20];                          //version


//global variables
double D0_MIN;                             //for d0
double Lnorm;                              //normalization length
double score_d8, d0, d0_search, dcu0;      //for TMscore search
double **score;            			       //Input score table for dynamic programming
bool   **path;                             //for dynamic programming
double **val;                              //for dynamic programming
int    xlen, ylen, minlen;                 //length of proteins
double **xa, **ya;                         //for input vectors xa[0...xlen-1][0..2], ya[0...ylen-1][0..2]
//in general, ya is regarded as native structure --> superpose xa onto ya
int    *xresno, *yresno;                   //residue numbers, used in fragment gapless threading
double **xtm, **ytm;                       //for TMscore search engine
double **xt;                               //for saving the superposed version of r_1 or xtm
char   *seqx, *seqy;                       //for the protein sequence
int    *secx, *secy;                       //for the secondary structure
double **r1, **r2;                         //for Kabsch rotation
double t[3], u[3][3];                      //Kabsch translation vector and rotation matrix

//argument variables
char out_reg[MAXLEN];
double Lnorm_ass, Lnorm_d0, d0_scale, d0A, d0B, d0u, d0a;
bool o_opt, a_opt, u_opt, d_opt, v_opt;
double TM3, TM4, TM5;
double d0_0;

#include "basic_fun.h"
#include <vector>


void superpose(
               int x_len,
               int y_len,
               double t[3],
               double u[3][3],
			   char *filename
               );

void output_alignment(
                      double **x,
                      double **y,
                      int x_len,
                      int y_len,
                      int invmap[]
					  );

vector <int> save_alignment(
							int y_len,
							int invmap[]
							);

#include "NW.h"
#include "Kabsch.h"
#include "TMalign.h"

void parameter_set4search(int xlen, int ylen)
{
	//parameter initilization for searching: D0_MIN, Lnorm, d0, d0_search, score_d8
	D0_MIN = 0.5;
	dcu0 = 4.25;                       //update 3.85-->4.25

	Lnorm = getmin(xlen, ylen);        //normaliz TMscore by this in searching
	if (Lnorm <= 19)                    //update 15-->19
	{
		d0 = 0.168;                   //update 0.5-->0.168
	}
	else
	{
		d0 = (1.24*pow((Lnorm*1.0 - 15), 1.0 / 3) - 1.8);
	}
	D0_MIN = d0 + 0.8;              //this should be moved to above
	d0 = D0_MIN;                  //update: best for search


	d0_search = d0;
	if (d0_search>8) d0_search = 8;
	if (d0_search<4.5) d0_search = 4.5;


	score_d8 = 1.5*pow(Lnorm*1.0, 0.3) + 3.5; //remove pairs with dis>d8 during search & final
}

void parameter_set4final(double len)
{
	D0_MIN = 0.5;

	Lnorm = len;            //normaliz TMscore by this in searching
	if (Lnorm <= 21)
	{
		d0 = 0.5;
	}
	else
	{
		d0 = (1.24*pow((Lnorm*1.0 - 15), 1.0 / 3) - 1.8);
	}
	if (d0<D0_MIN) d0 = D0_MIN;

	d0_search = d0;
	if (d0_search>8) d0_search = 8;
	if (d0_search<4.5) d0_search = 4.5;

}


void parameter_set4scale(int len, double d_s)
{

	d0 = d_s;
	Lnorm = len;            //normaliz TMscore by this in searching

	d0_search = d0;
	if (d0_search>8) d0_search = 8;
	if (d0_search<4.5) d0_search = 4.5;

}

int i11 = 0;
char* substr(char* s, int num)
{
	char *b;
	b = new char[30];
	for (i11 = 0; i11<num; i11++)
	{
		b[i11] = s[i11];
	}
	b[i11] = '\0';
	return b;
}

pair<pair<double, int>, double> input(char *xname,
		char *yname,
		char *seqy1,
		double **ya1,
		char *seqx1,
		double **xa1,
		double t0[3],
		double u0[3][3],
		int *no_x, 
		int *no_y)
{	
	//vector<vector<int> > invmaps;
	/*********************************************************************************/
	/*                                load data                                      */ 
    /*********************************************************************************/
	load_PDB_allocate_memory(xname, yname);
	/*********************************************************************************/
	/*                                parameter set                                  */
	/*********************************************************************************/ 
	parameter_set4search(xlen, ylen);          //please set parameters in the function
	int simplify_step = 40;               //for similified search engine
	int score_sum_method = 8;                //for scoring method, whether only sum over pairs with dis<score_d8

	int i;
	int *invmap0 = new int[ylen + 1];
	int *invmap = new int[ylen + 1];
	double TM, TMmax = -1;
	for (i = 0; i<ylen; i++)
	{
		invmap0[i] = -1;
	}


	double ddcc = 0.4;
	if (Lnorm <= 40) ddcc = 0.1;   //Lnorm was setted in parameter_set4search


	/*********************************************************************************/
	/*         get initial alignment with gapless threading                          */
	/*********************************************************************************/
	get_initial(xa, ya, xlen, ylen, invmap0);
	//find the max TMscore for this initial alignment with the simplified search_engin
	TM = detailed_search(xa, ya, xlen, ylen, invmap0, t, u, simplify_step, score_sum_method);
	if (TM>TMmax)
	{
		TMmax = TM;
	}
	//run dynamic programing iteratively to find the best alignment
	TM = DP_iter(xa, ya, xlen, ylen, t, u, invmap, 0, 2, 30);
	if (TM>TMmax)
	{
		TMmax = TM;
		for (int i = 0; i<ylen; i++)
		{
			invmap0[i] = invmap[i];
		}
	}

	//printf("\n>alignment based on gapless threading %.3f %.3f\n", TM, TMmax);
	//output_alignment(xa, ya, xlen, ylen,invmap0);



	/*********************************************************************************/
	/*         get initial alignment based on secondary structure                    */
	/*********************************************************************************/
	get_initial_ss(xa, ya, xlen, ylen, invmap);
	TM = detailed_search(xa, ya, xlen, ylen, invmap, t, u, simplify_step, score_sum_method);
	if (TM>TMmax)
	{
		TMmax = TM;
		for (int i = 0; i<ylen; i++)
		{
			invmap0[i] = invmap[i];
		}
	}
	if (TM > TMmax*0.2)
	{
		TM = DP_iter(xa, ya, xlen, ylen, t, u, invmap, 0, 2, 30);
		if (TM>TMmax)
		{
			TMmax = TM;
			for (int i = 0; i<ylen; i++)
			{
				invmap0[i] = invmap[i];
			}
		}
	}
	//output_align(invmap0, ylen);



	/*********************************************************************************/
	/*         get initial alignment based on local superposition                    */
	/*********************************************************************************/
	//=initial5 in original TM-align
	if (get_initial_local(xa, ya, xlen, ylen, invmap))
	{
		TM = detailed_search(xa, ya, xlen, ylen, invmap, t, u, simplify_step, score_sum_method);
		if (TM>TMmax)
		{
			TMmax = TM;
			for (int i = 0; i<ylen; i++)
			{
				invmap0[i] = invmap[i];
			}
		}
		if (TM > TMmax*ddcc)
		{
			TM = DP_iter(xa, ya, xlen, ylen, t, u, invmap, 0, 2, 2);
			if (TM>TMmax)
			{
				TMmax = TM;
				for (int i = 0; i<ylen; i++)
				{
					invmap0[i] = invmap[i];
				}
			}
		}
		//output_align(invmap0, ylen);
	}
	else
	{
		cout << endl << endl << "Warning: initial alignment from local superposition fail!" << endl << endl << endl;
	}





	/*********************************************************************************/
	/*    get initial alignment based on previous alignment+secondary structure      */
	/*********************************************************************************/
	//=initial3 in original TM-align
	get_initial_ssplus(xa, ya, xlen, ylen, invmap0, invmap);
	TM = detailed_search(xa, ya, xlen, ylen, invmap, t, u, simplify_step, score_sum_method);
	if (TM>TMmax)
	{
		TMmax = TM;
		for (i = 0; i<ylen; i++)
		{
			invmap0[i] = invmap[i];
		}
	}
	if (TM > TMmax*ddcc)
	{
		TM = DP_iter(xa, ya, xlen, ylen, t, u, invmap, 0, 2, 30);
		if (TM>TMmax)
		{
			TMmax = TM;
			for (i = 0; i<ylen; i++)
			{
				invmap0[i] = invmap[i];
			}
		}
	}
	else
	{
		//printf("\n>alignment based on secondary structure and previous alignment %.3f %.3f\n", TM, TMmax);
        //output_alignment(xa, ya, xlen, ylen,invmap);
	}
	//output_align(invmap0, ylen);






	/*********************************************************************************/
	/*        get initial alignment based on fragment gapless threading              */
	/*********************************************************************************/
	//=initial4 in original TM-align
	get_initial_fgt(xa, ya, xlen, ylen, xresno, yresno, invmap);
	TM = detailed_search(xa, ya, xlen, ylen, invmap, t, u, simplify_step, score_sum_method);
	if (TM>TMmax)
	{
		TMmax = TM;
		for (i = 0; i<ylen; i++)
		{
			invmap0[i] = invmap[i];
		}
	}
	if (TM > TMmax*ddcc)
	{
		TM = DP_iter(xa, ya, xlen, ylen, t, u, invmap, 1, 2, 2);
		if (TM>TMmax)
		{
			TMmax = TM;
			for (i = 0; i<ylen; i++)
			{
				invmap0[i] = invmap[i];
			}
		}
	}
	//output_align(invmap0, ylen);







	//*********************************************************************************//
	//     The alignment will not be changed any more in the following                 //
	//*********************************************************************************//
	//check if the initial alignment is generated approately
	bool flag = false;
	for (i = 0; i<ylen; i++)
	{
		if (invmap0[i] >= 0)
		{
			flag = true;
			break;
		}
	}
	if (!flag)
	{
		cout << "There is no alignment between the two proteins!" << endl;
		cout << "Program stop with no result!" << endl;
		
	}
	//cout << "final alignment" << endl;
	//output_align(invmap0, ylen);


	//*********************************************************************************//
	//       Detailed TMscore search engine  --> prepare for final TMscore             //
	//*********************************************************************************//
	//run detailed TMscore search engine for the best alignment, and
	//extract the best rotation matrix (t, u) for the best alginment
	simplify_step = 1;
	score_sum_method = 8;
	TM = detailed_search(xa, ya, xlen, ylen, invmap0, t, u, simplify_step, score_sum_method);

	//select pairs with dis<d8 for final TMscore computation and output alignment
	int n_ali8, k = 0;
	int n_ali = 0;
	int *m1, *m2;
	double d;
	m1 = new int[xlen]; //alignd index in x
	m2 = new int[ylen]; //alignd index in y
	do_rotation(xa, xt, xlen, t, u);
	k = 0;
	for (int j = 0; j<ylen; j++)
	{
		i = invmap0[j];
		if (i >= 0)//aligned
		{
			n_ali++;
			d = sqrt(dist(&xt[i][0], &ya[j][0]));
			if (d <= score_d8)
			{
				m1[k] = i;
				m2[k] = j;

				xtm[k][0] = xa[i][0];
				xtm[k][1] = xa[i][1];
				xtm[k][2] = xa[i][2];

				ytm[k][0] = ya[j][0];
				ytm[k][1] = ya[j][1];
				ytm[k][2] = ya[j][2];

				k++;
			}
		}
	}
	n_ali8 = k;
	//*********************************************************************************//
	//                               Final TMscore                                     //
	//                     Please set parameters for output                            //
	//*********************************************************************************//
	double rmsd, TM1, TM2;
	double d0_out = 5.0;
	simplify_step = 1;
	score_sum_method = 0;

	double TM_0, TM_1, TM_2;
	double Lnorm_0;
	
	
	if (xlen >= ylen)
	{
		Lnorm_0=ylen;
		parameter_set4final(Lnorm_0);
		d0_0 = d0;
		TM_0 = TMscore8_search(xtm, ytm, n_ali8, t0, u0, simplify_step, score_sum_method, &rmsd);
	}

	if (xlen < ylen)
	{
		Lnorm_0=xlen;
		parameter_set4final(Lnorm_0);
		d0_0 = d0;
		TM_0 = TMscore8_search(xtm, ytm, n_ali8, t0, u0, simplify_step, score_sum_method, &rmsd);
	}

	int align_len=output_results(xname, yname, xlen, ylen, t0, u0, rmsd, d0_out, m1, m2, n_ali8, n_ali, Lnorm_0,no_x,no_y);

/*	
	parameter_set4final(ylen);
	TM_1 = TMscore8_search(xtm, ytm, n_ali8, t0, u0, simplify_step, score_sum_method, &rmsd);
	parameter_set4final(xlen);
	TM_2 = TMscore8_search(xtm, ytm, n_ali8, t0, u0, simplify_step, score_sum_method, &rmsd);
*/
	
	for(int i=0;i<ylen;i++)
	{
		seqy1[i]=seqy[i];
	}
	for(int i=0;i<xlen;i++)
	{
		seqx1[i]=seqx[i];
	}
	for(int i=0;i<ylen;i++)
	{
		for(int j=0;j<3;j++)
		{
			ya1[i][j]=ya[i][j];
		}
	}
	for(int i=0;i<xlen;i++)
	{
		for(int j=0;j<3;j++)
		{
			xa1[i][j]=xa[i][j];
		}
	}
	
	//*********************************************************************************//
	//                            Done! Free memory                                    //
	//*********************************************************************************//
	free_memory();
	delete [] invmap0;
	delete [] invmap;
	delete [] m1;
	delete [] m2;
	//invmaps.clear();
	//invmaps_nr.clear();
	//invmaps_sorted.clear();	
	//cout<<TM_0<<endl;
	//pair<double, int> ans((TM_1+TM_2)/2, align_len);
	pair<double, int> ans1(TM_0, align_len);
	pair<pair<double, int>, double> ans2(ans1, rmsd);
	return ans2;
}

void get_u_t(double **xtm, double **ytm, int k1, double t1[3], double u1[3][3], int simplify_step, int score_sum_method, double *Rcomm, double d00)
{
    parameter_set4final(d00);

    TMscore8_search(xtm, ytm, k1, t1, u1, 1, 0, Rcomm);
    //cout<<t1[0]<<" "<<u1[0][0]<<" "<<u1[0][1]<<" "<<u1[0][2]<<endl;
}