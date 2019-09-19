/*
===============================================================================
   Implementation of TM-align in C/C++   

   This program is written by Jianyi Yang at
   Yang Zhang lab
   Center for Computational Medicine and Bioinformatics 
   University of Michigan 
   100 Washtenaw Avenue, Ann Arbor, MI 48109-2218 
                                                       
           
   Please report bugs and questions to yangji@umich.edu or or zhng@umich.edu
===============================================================================
*/
#ifndef _TMALIGN_H_
#define _TMALIGN_H_
#endif

void load_PDB_allocate_memory(char *xname, char *yname)
{    
    //------get length first------>
	xlen=get_PDB_len(xname);
	ylen=get_PDB_len(yname);
	minlen=min(xlen, ylen);

    //------allocate memory for x and y------>
	NewArray(&xa, xlen, 3);
    seqx   = new char[xlen+1];
	secx   = new int[xlen];
	xresno = new int[xlen];

	NewArray(&ya, ylen, 3);
	seqy    = new char[ylen+1];
	yresno  = new int[ylen];
	secy    = new int[ylen];

    
    
    //------load data------>  
    read_PDB(xname, xa, seqx, xresno);
    read_PDB(yname, ya, seqy, yresno);
    
    
    //------allocate memory for other temporary varialbes------>
 	NewArray(&r1, minlen, 3);
	NewArray(&r2, minlen, 3);
	NewArray(&xtm, minlen, 3);
	NewArray(&ytm, minlen, 3);
	NewArray(&xt, xlen, 3);

	NewArray(&score, xlen+1, ylen+1);
	NewArray(&path, xlen+1, ylen+1);
	NewArray(&val, xlen+1, ylen+1);  	
}


void free_memory()
{
	DeleteArray(&path, xlen+1);
	DeleteArray(&val, xlen+1);
	DeleteArray(&score, xlen+1);
	DeleteArray(&xa, xlen);
	DeleteArray(&xt, xlen);
	DeleteArray(&ya, ylen);
	DeleteArray(&r1, minlen);
	DeleteArray(&r2, minlen);
	DeleteArray(&xtm, minlen);
	DeleteArray(&ytm, minlen);
    
    
    delete [] seqx;
    delete [] seqy;
    delete [] secx;
    delete [] secy;
    delete [] xresno;
    delete [] yresno;
    
}


//     1, collect those residues with dis<d;
//     2, calculate TMscore
int score_fun8( double **xa, 
                double **ya, 
                int n_ali,
                double d,
                int i_ali[], 
                double *score1,
                int score_sum_method
              )
{
    double score_sum=0, di;
    double d_tmp=d*d;
	double d02=d0*d0;
	double score_d8_cut = score_d8*score_d8;
    
    int i, n_cut, inc=0;

    while(1)
    {
		n_cut=0;
        score_sum=0;
        for(i=0; i<n_ali; i++)
        {
            di = dist(xa[i], ya[i]);
            if(di<d_tmp)
            {
                i_ali[n_cut]=i;
                n_cut++;
            }
            if(score_sum_method==8)
            {				
                if(di<=score_d8_cut)
                {					
                    score_sum += 1/(1+di/d02);
                }                
            }
            else
            {
				score_sum += 1/(1+di/d02);
            }
        }
        //there are not enough feasible pairs, reliefe the threshold 		
        if(n_cut<3 && n_ali>3)
        {
			inc++;
			double dinc=(d+inc*0.5);
			d_tmp = dinc * dinc;
        }
        else
        {
            break;
        }

    }  

    *score1=score_sum/Lnorm;

    return n_cut;
}

// TMscore search engine
// input:   two aligned vector sets: x, y
//          scale parameter d0
//          simplify_step: 1 or 40 or other integers
//          score_sum_method: 0 for score over all pairs
//                            8 for socre over the pairs with dist<score_d8
//                                  
//          
// output:  the best rotaion matrix t0, u0 that results in highest TMscore
double TMscore8_search( double **xtm, 
                        double **ytm,
                        int Lali, 
                        double t0[3],
                        double u0[3][3],
                        int simplify_step,
                        int score_sum_method,
                        double *Rcomm
                       )
{   
    int i, m;
    double score_max, score, rmsd;    
    const int kmax=Lali;    
    int k_ali[1000], ka, k;
    double t[3];
    double u[3][3];
	double d;
	double **r11, **r21, **xt1;
    NewArray(&r11, Lali, 3);
    NewArray(&r21, Lali, 3);
    NewArray(&xt1, Lali, 3);
	//iterative parameters
	int n_it=20;            //maximum number of iterations
    const int n_init_max=6; //maximum number of different fragment length 
    int L_ini[n_init_max];  //fragment lengths, Lali, Lali/2, Lali/4 ... 4   
    int L_ini_min=4;
    if(Lali<4) L_ini_min=Lali;   
    int n_init=0, i_init;      
    for(i=0; i<n_init_max-1; i++)
    {
        n_init++;
        L_ini[i]=(int) (Lali/pow(2.0, (double) i));
        if(L_ini[i]<=L_ini_min)
        {
            L_ini[i]=L_ini_min;
            break;
        }
    }
    if(i==n_init_max-1)
    {
        n_init++;
        L_ini[i]=L_ini_min;
    }
    
    score_max=-1;
    //find the maximum score starting from local structures superposition
    int i_ali[10000], n_cut;
    int L_frag; //fragment length
    int iL_max; //maximum starting postion for the fragment
    for(i_init=0; i_init<n_init; i_init++)
    {
        L_frag=L_ini[i_init];
        iL_max=Lali-L_frag;
      
        i=0;   
        while(1)
        {
            //extract the fragment starting from position i 
            ka=0;
            for(k=0; k<L_frag; k++)
            {
				int kk=k+i;
                r11[k][0]=xtm[kk][0];  
                r11[k][1]=xtm[kk][1]; 
                r11[k][2]=xtm[kk][2];   
                
                r21[k][0]=ytm[kk][0];  
                r21[k][1]=ytm[kk][1]; 
                r21[k][2]=ytm[kk][2];
                
                k_ali[ka]=kk;
                ka++;
            }

            //extract rotation matrix based on the fragment
            Kabsch(r11, r21, L_frag, 1, &rmsd, t, u);
            //cout<<11<<endl;
            if(i_init==0)
            {
                *Rcomm=sqrt(rmsd/Lali);
            }

            do_rotation(xtm, xt1, Lali, t, u);

            //get subsegment of this fragment
            d=d0_search-1;
            n_cut=score_fun8(xt1, ytm, Lali, d, i_ali, &score, score_sum_method);
            if(score>score_max)
            {
                score_max=score;
                
                //save the rotation matrix
                for(k=0; k<3; k++)
                {
                    t0[k]=t[k];
                    u0[k][0]=u[k][0];
                    u0[k][1]=u[k][1];
                    u0[k][2]=u[k][2];
                }
            }
            
            //try to extend the alignment iteratively            
            d=d0_search+1;
            for(int it=0; it<n_it; it++)            
            {
                ka=0;
                for(k=0; k<n_cut; k++)
                {
                    m=i_ali[k];
                    r11[k][0]=xtm[m][0];  
                    r11[k][1]=xtm[m][1]; 
                    r11[k][2]=xtm[m][2];
                    
                    r21[k][0]=ytm[m][0];  
                    r21[k][1]=ytm[m][1]; 
                    r21[k][2]=ytm[m][2];
                    
                    k_ali[ka]=m;
                    ka++;
                } 
                //extract rotation matrix based on the fragment                
                Kabsch(r11, r21, n_cut, 1, &rmsd, t, u);
                do_rotation(xtm, xt1, Lali, t, u);
                n_cut=score_fun8(xt1, ytm, Lali, d, i_ali, &score, score_sum_method);
                if(score>score_max)
                {
                    score_max=score;

                    //save the rotation matrix
                    for(k=0; k<3; k++)
                    {
                        t0[k]=t[k];
                        u0[k][0]=u[k][0];
                        u0[k][1]=u[k][1];
                        u0[k][2]=u[k][2];
                    }                     
                }
                
                //check if it converges                 
			
                if(n_cut==ka)
                {				
                    for(k=0; k<n_cut; k++)
                    {
                        if(i_ali[k]!=k_ali[k])
						{
							break;
						}
                    }
                    if(k==n_cut)
                    {						
                        break; //stop iteration
                    }
                }                                                               
            } //for iteration            

			if(i<iL_max)
			{
				i=i+simplify_step; //shift the fragment		
				if(i>iL_max) i=iL_max;  //do this to use the last missed fragment
			}
			else if(i>=iL_max)
			{
				break;
			}
        }//while(1)
        //end of one fragment
    }//for(i_init
    DeleteArray(&r11, Lali);
    DeleteArray(&r21, Lali);
    DeleteArray(&xt1, Lali);
    return score_max;
}

//Comprehensive TMscore search engine
// input:   two vector sets: x, y
//          an alignment invmap0[] between x and y
//          simplify_step: 1 or 40 or other integers
//          score_sum_method: 0 for score over all pairs
//                            8 for socre over the pairs with dist<score_d8          
// output:  the best rotaion matrix t, u that results in highest TMscore
double detailed_search( double **x,
                        double **y, 
                        int x_len, 
                        int y_len, 
                        int invmap0[],
                        double t[3],
                        double u[3][3],
                        int simplify_step,
                        int score_sum_method                        
                       )
{
    //x is model, y is template, try to superpose onto y
    int i, j, k;     
    double tmscore;
    double rmsd;



    k=0;
    for(i=0; i<y_len; i++) 
    {
        j=invmap0[i];
        if(j>=0) //aligned
        {
            xtm[k][0]=x[j][0];
            xtm[k][1]=x[j][1];
            xtm[k][2]=x[j][2];
                
            ytm[k][0]=y[i][0];
            ytm[k][1]=y[i][1];
            ytm[k][2]=y[i][2];
            k++;
        }
    }
    
    //detailed search 40-->1
    tmscore=TMscore8_search(xtm, ytm, k, t, u, simplify_step, score_sum_method, &rmsd);  
   
    return tmscore;
}



//compute the score quickly in three iterations
double get_score_fast(double **x, double **y, int x_len, int y_len, int invmap[])
{
    double rms, tmscore, tmscore1, tmscore2;
    int i, j, k;

    k=0;
    for(j=0; j<y_len; j++)
    {
        i=invmap[j];
        if(i>=0)
        {
            r1[k][0]=x[i][0];
            r1[k][1]=x[i][1];
            r1[k][2]=x[i][2];

            r2[k][0]=y[j][0];
            r2[k][1]=y[j][1];
            r2[k][2]=y[j][2];
            
            xtm[k][0]=x[i][0];
            xtm[k][1]=x[i][1];
            xtm[k][2]=x[i][2];
            
            ytm[k][0]=y[j][0];
            ytm[k][1]=y[j][1];
            ytm[k][2]=y[j][2];                  
            
            k++;
        }
        else if(i!=-1)
        {
            string s="Wrong map!";
            char* c;
            const int len = s.length();
            c = new char[len+1];
            strcpy(c,s.c_str());
            PrintErrorAndQuit(c);
        }       
    }
    Kabsch(r1, r2, k, 1, &rms, t, u);
    
    //evaluate score   
    double di;
	const int len=k;
    double dis[10000];    
	double d00=d0_search;
	double d002=d00*d00;
	double d02=d0*d0;
	
    int n_ali=k;
	double xrot[3];
	tmscore=0;
	for(k=0; k<n_ali; k++)
	{
        transform(t, u, &xtm[k][0], xrot);        
        di=dist(xrot, &ytm[k][0]);
        dis[k]=di;
        tmscore += 1/(1+di/d02);
    }
	
   
   
   //second iteration 
    double d002t=d002;
    while(1)
    {
		j=0;
        for(k=0; k<n_ali; k++)
        {            
            if(dis[k]<=d002t)
            {
                r1[j][0]=xtm[k][0];
                r1[j][1]=xtm[k][1];
                r1[j][2]=xtm[k][2];
                
                r2[j][0]=ytm[k][0];
                r2[j][1]=ytm[k][1];
                r2[j][2]=ytm[k][2];
                
                j++;
            }
        }
        //there are not enough feasible pairs, relieve the threshold 
        if(j<3 && n_ali>3)
        {
            d002t += 0.5;
        }
        else
        {
            break;
        }
    }
    
    if(n_ali!=j)
    {
        Kabsch(r1, r2, j, 1, &rms, t, u);
    	tmscore1=0;
    	for(k=0; k<n_ali; k++)
    	{
            transform(t, u, &xtm[k][0], xrot);        
            di=dist(xrot, &ytm[k][0]);
            dis[k]=di;
            tmscore1 += 1/(1+di/d02);
        }
        
        //third iteration
        d002t=d002+1;
       
        while(1)
        {
			j=0;
            for(k=0; k<n_ali; k++)
            {            
                if(dis[k]<=d002t)
                {
                    r1[j][0]=xtm[k][0];
                    r1[j][1]=xtm[k][1];
                    r1[j][2]=xtm[k][2];
                    
                    r2[j][0]=ytm[k][0];
                    r2[j][1]=ytm[k][1];
                    r2[j][2]=ytm[k][2];
                                        
                    j++;
                }
            }
            //there are not enough feasible pairs, relieve the threshold 
            if(j<3 && n_ali>3)
            {
                d002t += 0.5;
            }
            else
            {
                break;
            }
        }

        //evaluate the score
        Kabsch(r1, r2, j, 1, &rms, t, u);
        tmscore2=0;
        for(k=0; k<n_ali; k++)
        {
            transform(t, u, &xtm[k][0], xrot);
            di=dist(xrot, &ytm[k][0]);
            tmscore2 += 1/(1+di/d02);
        }    
    }
    else
    {
        tmscore1=tmscore;
        tmscore2=tmscore;
    }
    
      
    if(tmscore1>=tmscore) tmscore=tmscore1;
    if(tmscore2>=tmscore) tmscore=tmscore2;

    
    return tmscore; // no need to normalize this score because it will not be used for latter scoring
}


//perform gapless threading to find the best initial alignment
//input: x, y, x_len, y_len
//output: y2x0 stores the best alignment: e.g., 
//y2x0[j]=i means:
//the jth element in y is aligned to the ith element in x if i>=0 
//the jth element in y is aligned to a gap in x if i==-1
double get_initial( double **x, 
                    double **y, 
                    int x_len,
                    int y_len, 
                    int *y2x
                   )
{
    int min_len=getmin(x_len, y_len);
    string s="Sequence is too short <=5!\n";
    if(min_len<=5) cout<<s<<endl;
    
    int min_ali= min_len/2;              //minimum size of considered fragment 
    if(min_ali<=5)  min_ali=5;    
    int n1, n2;
    n1 = -y_len+min_ali; 
    n2 = x_len-min_ali;

    int i, j, k, k_best;
    double tmscore, tmscore_max=-1;

    k_best=n1;
    for(k=n1; k<=n2; k++)
    {
        //get the map
        for(j=0; j<y_len; j++)
        {
            i=j+k;
            if(i>=0 && i<x_len)
            {
                y2x[j]=i;
            }
            else
            {
                y2x[j]=-1;
            }
        }
        
        //evaluate the map quickly in three iterations
		//this is not real tmscore, it is used to evaluate the goodness of the initial alignment
        tmscore=get_score_fast(x, y, x_len, y_len, y2x); 
        if(tmscore>=tmscore_max)
        {
            tmscore_max=tmscore;
            k_best=k;
        }
    }
    
    //extract the best map
    k=k_best;
    for(j=0; j<y_len; j++)
    {
        i=j+k;
        if(i>=0 && i<x_len)
        {
            y2x[j]=i;
        }
        else
        {
            y2x[j]=-1;
        }
    }    

    return tmscore_max;
}

void smooth(int *sec, int len)
{
	int i, j;
	//smooth single  --x-- => -----
	for(i=2; i<len-2; i++)
	{
		if(sec[i]==2 || sec[i]==4)
		{
			j=sec[i];
			if(sec[i-2] != j)
			{
				if(sec[i-1] != j)
				{
					if(sec[i+1] != j)
					{
						if(sec[i+2] != j) 
						{
							sec[i]=1;
						}
					}
				}
			}
		}
	}

	//   smooth double 
	//   --xx-- => ------

	for(i=0; i<len-5; i++)
	{
		//helix
		if(sec[i] != 2)
		{
			if(sec[i+1] != 2)
			{
				if(sec[i+2] == 2)
				{
					if(sec[i+3] == 2)
					{
						if(sec[i+4] != 2)
						{
							if(sec[i+5] != 2)
							{
								sec[i+2]=1;
								sec[i+3]=1;
							}
						}
					}
				}
			}
		}

		//beta
		if(sec[i] != 4)
		{
			if(sec[i+1] != 4)
			{
				if(sec[i+2] ==4)
				{
					if(sec[i+3] == 4)
					{
						if(sec[i+4] != 4)
						{
							if(sec[i+5] != 4)
							{
								sec[i+2]=1;
								sec[i+3]=1;
							}
						}
					}
				}
			}
		}
	}

	//smooth connect
	for(i=0; i<len-2; i++)
	{		
		if(sec[i] == 2)
		{
			if(sec[i+1] != 2)
			{
				if(sec[i+2] == 2)
				{
					sec[i+1]=2;
				}
			}
		}
		else if(sec[i] == 4)
		{
			if(sec[i+1] != 4)
			{
				if(sec[i+2] == 4)
				{
					sec[i+1]=4;
				}
			}
		}
	}

}

int sec_str(double dis13, double dis14, double dis15, double dis24, double dis25, double dis35)
{
	int s=1;
	
	double delta=2.1;
	if(fabs(dis15-6.37)<delta)
	{
		if(fabs(dis14-5.18)<delta)
		{
			if(fabs(dis25-5.18)<delta)
			{
				if(fabs(dis13-5.45)<delta)
				{
					if(fabs(dis24-5.45)<delta)
					{
						if(fabs(dis35-5.45)<delta)
						{
							s=2; //helix						
							return s;
						}
					}
				}
			}
		}
	}

	delta=1.42;
	if(fabs(dis15-13)<delta)
	{
		if(fabs(dis14-10.4)<delta)
		{
			if(fabs(dis25-10.4)<delta)
			{
				if(fabs(dis13-6.1)<delta)
				{
					if(fabs(dis24-6.1)<delta)
					{
						if(fabs(dis35-6.1)<delta)
						{
							s=4; //strand
							return s;
						}
					}
				}
			}
		}
	}

	if(dis15 < 8)
	{
		s=3; //turn
	}	  


	return s;
}


//1->coil, 2->helix, 3->turn, 4->strand
void make_sec(double **x, int len, int *sec)
{
    int j1, j2, j3, j4, j5;
	double d13, d14, d15, d24, d25, d35;
    for(int i=0; i<len; i++)
    { 	
		sec[i]=1;
        j1=i-2;
        j2=i-1;
        j3=i;
        j4=i+1;
        j5=i+2;		
        
        if(j1>=0 && j5<len)
		{
			d13=sqrt(dist(x[j1], x[j3]));
			d14=sqrt(dist(x[j1], x[j4]));
			d15=sqrt(dist(x[j1], x[j5]));
			d24=sqrt(dist(x[j2], x[j4]));
			d25=sqrt(dist(x[j2], x[j5]));
			d35=sqrt(dist(x[j3], x[j5]));
			sec[i]=sec_str(d13, d14, d15, d24, d25, d35);			
		}    
    } 
    //smooth(sec, len);
}




//get initial alignment from secondary structure alignment
//input: x, y, x_len, y_len
//output: y2x stores the best alignment: e.g., 
//y2x[j]=i means:
//the jth element in y is aligned to the ith element in x if i>=0 
//the jth element in y is aligned to a gap in x if i==-1
void get_initial_ss(  double **x, 
					  double **y, 
					  int x_len,
					  int y_len, 
					  int *y2x
                      )
{
    //assign secondary structures
	make_sec(x, x_len, secx);
	make_sec(y, y_len, secy);

	double gap_open=-1.0;
	NWDP_TM(secx, secy, x_len, y_len, gap_open, y2x);    
}


// get_initial5 in TMalign
//get initial alignment of local structure superposition
//input: x, y, x_len, y_len
//output: y2x stores the best alignment: e.g., 
//y2x[j]=i means:
//the jth element in y is aligned to the ith element in x if i>=0 
//the jth element in y is aligned to a gap in x if i==-1
bool get_initial_local(double **x,
		       double **y,
		       int x_len,
		       int y_len,
    int *y2x
		       )
{
  double GL, rmsd;
  double t[3];
  double u[3][3];

  double d01 = d0 + 1.5;
  if (d01 < D0_MIN) d01 = D0_MIN;
  double d02 = d01*d01;

  double GLmax = 0;
  int aL = getmin(x_len, y_len);
  int *invmap = new int[y_len + 1];

  // jump on sequence1-------------->
  int n_jump1 = 0;
  if (x_len > 250)
    n_jump1 = 45;
  else if (x_len > 200)
    n_jump1 = 35;
  else if (x_len > 150)
    n_jump1 = 25;
  else
    n_jump1 = 15;
  if (n_jump1 > (x_len / 3))
    n_jump1 = x_len / 3;

  // jump on sequence2-------------->
  int n_jump2 = 0;
  if (y_len > 250)
    n_jump2 = 45;
  else if (y_len > 200)
    n_jump2 = 35;
  else if (y_len > 150)
    n_jump2 = 25;
  else
    n_jump2 = 15;
  if (n_jump2 > (y_len / 3))
    n_jump2 = y_len / 3;
  // fragment to superimpose-------------->
  int n_frag[2] = { 20, 100 };
  if (n_frag[0] > (aL / 3))
    n_frag[0] = aL / 3;
  if (n_frag[1] > (aL / 2))
    n_frag[1] = aL / 2;

  // start superimpose search-------------->
  bool flag = false;
  for (int i_frag = 0; i_frag < 2; i_frag++)
    {
      int m1 = x_len - n_frag[i_frag] + 1;
      int m2 = y_len - n_frag[i_frag] + 1;

      //for (int i = 1; i<m1; i = i + n_jump1) //index starts from 0, different from FORTRAN
      // for debug
      for (int i = 0; i<m1; i = i + n_jump1) //index starts from 0, different from FORTRAN
        {
	  //for (int j = 1; j<m2; j = j + n_jump2)

	  for (int j = 0; j<m2; j = j + n_jump2)
            {
	      for (int k = 0; k<n_frag[i_frag]; k++) //fragment in y
                {
		  r1[k][0] = x[k + i][0];
		  r1[k][1] = x[k + i][1];
		  r1[k][2] = x[k + i][2];

		  r2[k][0] = y[k + j][0];
		  r2[k][1] = y[k + j][1];
		  r2[k][2] = y[k + j][2];
                }
	      // superpose the two structures and rotate it
	      Kabsch(r1, r2, n_frag[i_frag], 1, &rmsd, t, u);

	      double gap_open = 0.0;
	      NWDP_TM(x, y, x_len, y_len, t, u, d02, gap_open, invmap);
	      GL = get_score_fast(x, y, x_len, y_len, invmap);
	      if (GL>GLmax)
                {
		  GLmax = GL;
		  for (int ii = 0; ii<y_len; ii++)
                    {
		      y2x[ii] = invmap[ii];
                    }
		  flag = true;
                }
	    }
        }
    }
  delete[] invmap;
  return flag;
}








  

//with invmap(i) calculate score(i,j) using RMSD rotation
void score_matrix_rmsd(  double **x, 
						 double **y, 
						 int x_len,
						 int y_len,
						 int *y2x
						 )
{
	double t[3], u[3][3];
	double rmsd, dij;
	double d01=d0+1.5;
	if(d01 < D0_MIN) d01=D0_MIN;
	double d02=d01*d01;

	double xx[3];
	int i, k=0;
	for(int j=0; j<y_len; j++)
	{
		i=y2x[j];
		if(i>=0)
		{
			r1[k][0]=x[i][0];  
			r1[k][1]=x[i][1]; 
			r1[k][2]=x[i][2];   
            
			r2[k][0]=y[j][0];  
			r2[k][1]=y[j][1]; 
			r2[k][2]=y[j][2];
			
			k++;
		}
	}
	Kabsch(r1, r2, k, 1, &rmsd, t, u);
	//do_rotation(x, xt, x_len, t, u);
	
	
	
	for(int ii=0; ii<x_len; ii++)
	{		
		transform(t, u, &x[ii][0], xx);
		for(int jj=0; jj<y_len; jj++)
		{
			//dij=dist(&xt[ii][0], &y[jj][0]);   
			dij=dist(xx, &y[jj][0]); 
			score[ii+1][jj+1] = 1.0/(1+dij/d02);
			//	cout << ii+1 << " " << jj+1 << " " << score[ii+1][jj+1]<< endl;
		}
	}		
}


void score_matrix_rmsd_sec(  double **x, 
							 double **y, 
							 int x_len,
							 int y_len,
							 int *y2x
							 )
{
	double t[3], u[3][3];
	double rmsd, dij;
	double d01=d0+1.5;
	if(d01 < D0_MIN) d01=D0_MIN;
	double d02=d01*d01;

	double xx[3];
	int i, k=0;
	for(int j=0; j<y_len; j++)
	{
		i=y2x[j];
		if(i>=0)
		{
			r1[k][0]=x[i][0];  
			r1[k][1]=x[i][1]; 
			r1[k][2]=x[i][2];   
            
			r2[k][0]=y[j][0];  
			r2[k][1]=y[j][1]; 
			r2[k][2]=y[j][2];
			
			k++;
		}
	}
	Kabsch(r1, r2, k, 1, &rmsd, t, u);
	
	
	for(int ii=0; ii<x_len; ii++)
	{		
		transform(t, u, &x[ii][0], xx);
		for(int jj=0; jj<y_len; jj++)
		{
			dij=dist(xx, &y[jj][0]); 
			if(secx[ii]==secy[jj])
			{
				score[ii+1][jj+1] = 1.0/(1+dij/d02) + 0.5;
			}
			else
			{
				score[ii+1][jj+1] = 1.0/(1+dij/d02);
			}		
		}
	}
	

}

//get initial alignment from secondary structure and previous alignments
//input: x, y, x_len, y_len
//output: y2x stores the best alignment: e.g., 
//y2x[j]=i means:
//the jth element in y is aligned to the ith element in x if i>=0 
//the jth element in y is aligned to a gap in x if i==-1
void get_initial_ssplus( double **x, 
						 double **y, 
						 int x_len,
						 int y_len,
						 int *y2x0,
						 int *y2x						
						 )
{

	//create score matrix for DP
	score_matrix_rmsd_sec(x, y, x_len, y_len, y2x0);
	
	double gap_open=-1.0;
	NWDP_TM(x_len, y_len, gap_open, y2x);
}


void find_max_frag(double **x, int *resno, int len, int *start_max, int *end_max)
{
	int r_min, fra_min=4;           //minimum fragment for search
	double d;
	int start;
	int Lfr_max=0, flag;

	r_min= (int) (len*1.0/3.0); //minimum fragment, in case too small protein
	if(r_min > fra_min) r_min=fra_min;
	
	int inc=0;
	double dcu0_cut=dcu0*dcu0;;
	double dcu_cut=dcu0_cut;

	while(Lfr_max < r_min)
	{		
		Lfr_max=0;			
		int j=1;    //number of residues at nf-fragment
		start=0;
		for(int i=1; i<len; i++)
		{			
			d = dist(x[i-1], x[i]);
			flag=0;
			if(dcu_cut>dcu0_cut)
			{
				if(d<dcu_cut)
				{
					flag=1;
				}
			}
			else if(resno[i] == (resno[i-1]+1)) //necessary??
			{
				if(d<dcu_cut)
				{
					flag=1;
				}
			}

			if(flag==1)
			{
				j++;

				if(i==(len-1))
				{
					if(j > Lfr_max) 
					{
						Lfr_max=j;
						*start_max=start;
						*end_max=i;						
					}
					j=1;
				}
			}
			else
			{
				if(j>Lfr_max) 
				{
					Lfr_max=j;
					*start_max=start;
					*end_max=i-1;										
				}

				j=1;
				start=i;
			}
		}// for i;
		
		if(Lfr_max < r_min)
		{
			inc++;
			double dinc=pow(1.1, (double) inc) * dcu0;
			dcu_cut= dinc*dinc;
		}
	}//while <;	
}

//perform fragment gapless threading to find the best initial alignment
//input: x, y, x_len, y_len
//output: y2x0 stores the best alignment: e.g., 
//y2x0[j]=i means:
//the jth element in y is aligned to the ith element in x if i>=0 
//the jth element in y is aligned to a gap in x if i==-1
double get_initial_fgt( double **x, 
						double **y, 
						int x_len,
						int y_len, 
						int *xresno,
						int *yresno,
						int *y2x
						)
{
	int fra_min=4;           //minimum fragment for search
	int fra_min1=fra_min-1;  //cutoff for shift, save time

	int xstart=0, ystart=0, xend=0, yend=0;

	find_max_frag(x, xresno, x_len,  &xstart, &xend);
	find_max_frag(y, yresno, y_len, &ystart, &yend);


	int Lx = xend-xstart+1;
	int Ly = yend-ystart+1;
	int *ifr, *y2x_;
	int L_fr=getmin(Lx, Ly);
	ifr= new int[L_fr];
	y2x_= new int[y_len+1];

	//select what piece will be used (this may araise ansysmetry, but
	//only when L1=L2 and Lfr1=Lfr2 and L1 ne Lfr1
	//if L1=Lfr1 and L2=Lfr2 (normal proteins), it will be the same as initial1

	if(Lx<Ly || (Lx==Ly && x_len<=y_len))
	{		
		for(int i=0; i<L_fr; i++)
		{
			ifr[i]=xstart+i;
		}
	}
	else if(Lx>Ly || (Lx==Ly && x_len>y_len))
	{		
		for(int i=0; i<L_fr; i++)
		{
			ifr[i]=ystart+i;
		}	
	}

	
	int L0=getmin(x_len, y_len); //non-redundant to get_initial1
	if(L_fr==L0)
	{
		int n1= (int)(L0*0.1); //my index starts from 0
		int n2= (int)(L0*0.89);

		int j=0;
		for(int i=n1; i<= n2; i++)
		{
			ifr[j]=ifr[i];
			j++;
		}
		L_fr=j;
	}


	//gapless threading for the extracted fragment
	double tmscore, tmscore_max=-1;

	if(Lx<Ly || (Lx==Ly && x_len<=y_len))
	{
		int L1=L_fr;
	    int min_len=getmin(L1, y_len);    
		int min_ali= (int) (min_len/2.5);              //minimum size of considered fragment 
		if(min_ali<=fra_min1)  min_ali=fra_min1;    
		int n1, n2;
		n1 = -y_len+min_ali; 
		n2 = L1-min_ali;

		int i, j, k;
		for(k=n1; k<=n2; k++)
		{
			//get the map
			for(j=0; j<y_len; j++)
			{
				i=j+k;
				if(i>=0 && i<L1)
				{				
					y2x_[j]=ifr[i];
				}
				else
				{
					y2x_[j]=-1;
				}
			}

			//evaluate the map quickly in three iterations
			tmscore=get_score_fast(x, y, x_len, y_len, y2x_);

			if(tmscore>=tmscore_max)
			{
				tmscore_max=tmscore;
				for(j=0; j<y_len; j++)
				{
					y2x[j]=y2x_[j];
				}
			}
		}
	}
	else
	{
		int L2=L_fr;
	    int min_len=getmin(x_len, L2);    
		int min_ali= (int) (min_len/2.5);              //minimum size of considered fragment 
		if(min_ali<=fra_min1)  min_ali=fra_min1;    
		int n1, n2;
		n1 = -L2+min_ali; 
		n2 = x_len-min_ali;

		int i, j, k;	

		for(k=n1; k<=n2; k++)
		{
			//get the map
			for(j=0; j<y_len; j++)
			{
				y2x_[j]=-1;
			}

			for(j=0; j<L2; j++)
			{
				i=j+k;
				if(i>=0 && i<x_len)
				{
					y2x_[ifr[j]]=i;
				}
			}
        
			//evaluate the map quickly in three iterations
			tmscore=get_score_fast(x, y, x_len, y_len, y2x_);
			if(tmscore>=tmscore_max)
			{
				tmscore_max=tmscore;
				for(j=0; j<y_len; j++)
				{
					y2x[j]=y2x_[j];
				}
			}
		}
	}    


	delete [] ifr;
	delete [] y2x_;
    return tmscore_max;
}





//heuristic run of dynamic programing iteratively to find the best alignment
//input: initial rotation matrix t, u
//       vectors x and y, d0
//output: best alignment that maximizes the TMscore, will be stored in invmap
double DP_iter( double **x,
                double **y, 
                int x_len, 
                int y_len, 
                double t[3],
                double u[3][3],
				int invmap0[],
				int g1,
				int g2,
				int iteration_max                                   
				)
{
    double gap_open[2]={-0.6, 0};
    double rmsd; 
    int *invmap=new int[y_len+1];
    
    int iteration, i, j, k;
    double tmscore, tmscore_max, tmscore_old=0;    
    int score_sum_method=8, simplify_step=40;
    tmscore_max=-1;

	//double d01=d0+1.5;
    double d02=d0*d0;
    for(int g=g1; g<g2; g++)
    {
        for(iteration=0; iteration<iteration_max; iteration++)
        {           
			NWDP_TM(x, y, x_len, y_len, t, u, d02, gap_open[g], invmap);
            
            k=0;
            for(j=0; j<y_len; j++) 
            {
                i=invmap[j];

                if(i>=0) //aligned
                {
                    xtm[k][0]=x[i][0];
                    xtm[k][1]=x[i][1];
                    xtm[k][2]=x[i][2];
                    
                    ytm[k][0]=y[j][0];
                    ytm[k][1]=y[j][1];
                    ytm[k][2]=y[j][2];
                    k++;
                }
            }

            tmscore=TMscore8_search(xtm, ytm, k, t, u, simplify_step, score_sum_method, &rmsd);

           
            if(tmscore>tmscore_max)
            {
                tmscore_max=tmscore;
                for(i=0; i<y_len; i++) 
                {                
                    invmap0[i]=invmap[i];                                      
                }				
            }
    
            if(iteration>0)
            {
                if(fabs(tmscore_old-tmscore)<0.000001)
                {     
                    break;       
                }
            }
            tmscore_old=tmscore;
        }// for iteration           
        
    }//for gapopen
    
    
    delete []invmap;
    return tmscore_max;
}


void output_superpose(char *xname,
					  char *yname,
					  int x_len,
					  int y_len,
					  double t[3],
					  double u[3][3],
					  double rmsd,
					  double d0_out,
					  int m1[], 
					  int m2[],
					  int n_ali8,
					  double seq_id,
					  double TM_0,
					  double Lnorm_0,
					  double d0_0
					 )
{
	int i, j, j1;
	double dis2;

	int max=2000;


	//aligned region
	FILE *fp=fopen(out_reg, "w");
	fprintf(fp, "load inline\n");
	fprintf(fp, "select atomno<%d\n", max);
	fprintf(fp, "wireframe .45\n");
	fprintf(fp, "select none\n");
	fprintf(fp, "select atomno>%d\n", max);
	fprintf(fp, "wireframe .20\n");
	fprintf(fp, "color white\n");


	do_rotation(xa, xt, x_len, t, u);
	for(i=0; i<n_ali8; i++)
	{
		j=m1[i];
		j1=m2[i];
		dis2=sqrt(dist(&xt[j][0], &ya[j1][0])) ;
		if(dis2<=d0_out)
		{
			fprintf(fp, "select atomno=%d\n", j+1);
			fprintf(fp, "color red\n");

			fprintf(fp, "select atomno=%d\n", max+j1+1);
			fprintf(fp, "color red\n");
		}
	}

	fprintf(fp, "select all\n");
	fprintf(fp, "exit\n");
	fprintf(fp, "REMARK TM-align Version %s\n", version);
	fprintf(fp, "REMARK Structure A:%s   Size=%d\n", yname, y_len);
	fprintf(fp, "REMARK Structure B:%s   Size=%d\n", xname, x_len);
	fprintf(fp, "REMARK TM-score is normalized by %.2f, d0=%.2f\n",  Lnorm_0, d0_0);
	fprintf(fp, "REMARK Aligned length=%d, RMSD=%.2f, TM-score=%.5f, ID=%.3f\n", n_ali8, rmsd, TM_0, seq_id);


	char AA[3];
	//superposed structure B
	for(i=0; i<n_ali8; i++)
	{
		j=m1[i];
		AAmap3(seqx[j], AA);
		fprintf(fp, "ATOM  %5d  CA  %3s %5d    %8.3f%8.3f%8.3f\n", j+1, AA, xresno[j], xt[j][0], xt[j][1], xt[j][2]);
	}
	fprintf(fp, "TER\n");

	for(i=1; i<n_ali8; i++)
	{
		j=m1[i-1]+1;
		j1=m1[i]+1;
		fprintf(fp, "CONECT%5d%5d\n", j, j1);
	}
	//structure A
	for(i=0; i<n_ali8; i++)
	{
		j=m2[i];
		AAmap3(seqy[j], AA);
		fprintf(fp, "ATOM  %5d  CA  %3s %5d    %8.3f%8.3f%8.3f\n", j+max+1, AA, yresno[j], ya[j][0], ya[j][1], ya[j][2]);
	}
	fprintf(fp, "TER\n");
	for(i=1; i<n_ali8; i++)
	{
		j=max+m2[i-1]+1;
		j1=max+m2[i]+1;
		fprintf(fp, "CONECT%5d%5d\n", j, j1);
	}

	fclose(fp);








	//all regions
	char str[3000];
	sprintf(str, "%s_all", out_reg);
	fp=fopen(str, "w");
	fprintf(fp, "load inline\n");
	fprintf(fp, "select atomno<%d\n", max);
	fprintf(fp, "wireframe .45\n");
	fprintf(fp, "select none\n");
	fprintf(fp, "select atomno>%d\n", max);
	fprintf(fp, "wireframe .20\n");
	fprintf(fp, "color white\n");

	for(i=0; i<n_ali8; i++)
	{
		j=m1[i];
		j1=m2[i];
		dis2=sqrt(dist(&xt[j][0], &ya[j1][0])) ;
		if(dis2<=d0_out)
		{
			fprintf(fp, "select atomno=%d\n", j+1);
			fprintf(fp, "color red\n");

			fprintf(fp, "select atomno=%d\n", max+j1+1);
			fprintf(fp, "color red\n");
		}
	}


	fprintf(fp, "select all\n");
	fprintf(fp, "exit\n");
	fprintf(fp, "REMARK TM-align Version %s\n", version);
	fprintf(fp, "REMARK Structure A:%s   Size=%d\n", yname, y_len);
	fprintf(fp, "REMARK Structure B:%s   Size=%d\n", xname, x_len);
	fprintf(fp, "REMARK TM-score is normalized by %.2f, d0=%.2f\n",  Lnorm_0, d0_0);
	fprintf(fp, "REMARK Aligned length=%d, RMSD=%.2f, TM-score=%.5f, ID=%.3f\n", n_ali8, rmsd, TM_0, seq_id);



	//superposed structure B
	for(i=0; i<x_len; i++)
	{
		j=i;
		AAmap3(seqx[j], AA);
		fprintf(fp, "ATOM  %5d  CA  %3s %5d    %8.3f%8.3f%8.3f\n", j+1, AA, xresno[j], xt[j][0], xt[j][1], xt[j][2]);
	}
	fprintf(fp, "TER\n");

	for(i=1; i<x_len; i++)
	{
		fprintf(fp, "CONECT%5d%5d\n", i, i+1);
	}
	//structure A
	for(i=0; i<y_len; i++)
	{
		j=i;
		AAmap3(seqy[j], AA);
		fprintf(fp, "ATOM  %5d  CA  %3s %5d    %8.3f%8.3f%8.3f\n", j+max+1, AA, yresno[j], ya[j][0], ya[j][1], ya[j][2]);
	}
	fprintf(fp, "TER\n");
	for(i=1; i<y_len; i++)
	{
		fprintf(fp, "CONECT%5d%5d\n", max+i, max+i+1);
	}

	fclose(fp);
}
int kk=0;
char *seqM, *seqxA, *seqyA;
//output the final results
int output_results(char *xname,
    char *yname,
    int x_len,
    int y_len,
    double t[3],
    double u[3][3],
    double rmsd,
    double d0_out,
    int m1[],
    int m2[],
    int n_ali8,
    int n_ali,
    double Lnorm_0,
	int *no_x,
	int *no_y
    )
{
    double seq_id;
    int i, j, k;
    double d;
    int ali_len = x_len + y_len; //maximum length of alignment

    seqM = new char[ali_len];
    seqxA = new char[ali_len];
    seqyA = new char[ali_len];
	

    do_rotation(xa, xt, x_len, t, u);
    kk=0;
    seq_id = 0;
    int i_old = 0, j_old = 0;
    for (k = 0; k < n_ali8; k++)
    {
        for (i = i_old; i < m1[k]; i++)
        {
            //align x to gap
            seqxA[kk] = seqx[i];
            seqyA[kk] = '-';
            seqM[kk] = ' ';
            kk++;
        }

        for (j = j_old; j < m2[k]; j++)
        {
            //align y to gap
            seqxA[kk] = '-';
            seqyA[kk] = seqy[j];
            seqM[kk] = ' ';
            kk++;
        }

        seqxA[kk] = seqx[m1[k]];
        seqyA[kk] = seqy[m2[k]];
        if (seqxA[kk] == seqyA[kk])
        {
            seq_id++;
        }
        d = sqrt(dist(&xt[m1[k]][0], &ya[m2[k]][0]));
        if (d < d0_out)
        {
            seqM[kk] = ':';
        }
        else
        {
            seqM[kk] = '.';
        }
        kk++;
        i_old = m1[k] + 1;
        j_old = m2[k] + 1;
    }

    //tail
    for (i = i_old; i < x_len; i++)
    {
        //align x to gap
        seqxA[kk] = seqx[i];
        seqyA[kk] = '-';
        seqM[kk] = ' ';
        kk++;
    }
    for (j = j_old; j < y_len; j++)
    {
        //align y to gap
        seqxA[kk] = '-';
        seqyA[kk] = seqy[j];
        seqM[kk] = ' ';
        kk++;
    }

	
    seqxA[kk] = '\0';
    seqyA[kk] = '\0';
    seqM[kk] = '\0';
	
	int count=0;
	for(i=0;i<kk;i++)
	{
		if(seqxA[i]=='-')
		{
			no_x[i]=-1;
		}
		else
		{
			no_x[i]=count;
			count++;
		}
	}
	
	count=0;
	for(j=0;j<kk;j++)
	{
		if(seqyA[j]=='-')
		{
			no_y[j]=-1;
		}
		else
		{
			no_y[j]=count;
			count++;
		}
	}
	no_x[kk]='\0';
	no_y[kk]='\0';
	delete [] seqM;
	delete [] seqxA;
	delete [] seqyA;
	
	return kk;
	
    seq_id = seq_id / (n_ali8 + 0.00000001); //what did by TMalign, but not reasonable, it should be n_ali8    
}
