#include "TMM.h"
#include "TMalign_main.h"

vector<char*> Iname;	//save Initial structure name
vector<int> Ilen;	//save Initial structure length
vector<vector<char> > Iseq;	//save Initial structure residue sequence queue
vector<vector<vector<double> > > Ico;	//save Initial structure coordinate

vector<string> Ename;	//save Element name
vector<int> Elen;	//save Element len
vector<vector<int> > Eno;	//save Element order

vector<pair<string, string> > Pname;	//save pairs name 
vector<int> Palen;	//save pairs aligned len
vector<vector<int> > Pno1;	//save structure 1 order of pairs  
vector<vector<int> > Pno2;	//save structure 2 order of pairs
vector<vector<vector<double> > > Pscore;	//save pairs score matrix
vector<vector<vector<double> > > Pdis;
vector<vector<double> > Pt;
vector<vector<vector<double> > > Pu;
vector<double> Pd0;
vector<double> vec_TM;

int cor;
int *full_len;	//save structure full-atom length 
int *conect_len;
double ***full_co;
char ***num1;
char ***fla;
char ***ato;
char ***res;
char ***num2;
string **conect;
vector<vector<double> > vec_dis;
vector<vector<vector<double> > > new_co;

void //output matrix.txt and save information
TMM::matrix_output(vector<char*> vec_input)	//vector of input structure name
{
	tot_num = vec_input.size();	//total number of structures
	
	char *tmp1;
	for(int i=0; i< tot_num-1; i++)
	{
		for(int j=i+1; j< tot_num; j++)
		{
			if(strcmp(vec_input[i], vec_input[j]) > 0)
			{
				tmp1 = vec_input[i];
				vec_input[i] = vec_input[j];
				vec_input[j] = tmp1; 
			}
		}
	}
	ofstream osfile;	//output stream
	osfile.open("matrix.txt");
	FILE*fp=NULL;
	fp=fopen("pairwise_TMscore.txt","w"); 
	FILE*fp1=NULL;
	fp1=fopen("pairwise_rmsd.txt","w"); 
	FILE*fp2=NULL;
	fp2=fopen("infile","w"); 
	
	double **distance_matrix;	//distance matrix
	double **rmsd_matrix;
	NewArray(&distance_matrix, tot_num, tot_num);
	NewArray(&rmsd_matrix, tot_num, tot_num);

	for (int i=0; i<tot_num; i++)	//initialize
	{
		for (int j=0; j<tot_num; j++)
		{
			distance_matrix[i][j] = 0;
		}
	}

	for (int i=0; i<tot_num; i++)	//initialize
	{
		for (int j=0; j<tot_num; j++)
		{
			rmsd_matrix[i][j] = 0;
		}
	}

	for (int i=0; i<tot_num; i++)	//save information
	{
		for (int j=0; j<i; j++)
		{
			int len_x = get_PDB_len(vec_input[j]);	//length of structure x 
			int len_y = get_PDB_len(vec_input[i]);	//length of structure y
			char *seq_x = new char[len_x];	//sequence of structure x
			char *seq_y = new char[len_y];	//sequence of structure y
			double **co_x,**co_y;	//coordinate of structure x and y
			NewArray(&co_x, len_x, 3);
			NewArray(&co_y, len_y, 3);
			vector<vector<double> > co_xv(len_x, vector<double>(3));	//save coordinate of structure x in vector
			vector<vector<double> > co_yv(len_y, vector<double>(3));	//save coordinate of structure y in vector
			double t[3], u[3][3];	//shift vector t, rotation matrix u
			int *no_x = new int [len_x+len_y];	//order of structure x
			int *no_y = new int [len_x+len_y];	//order of structure y
			int align_len;
			pair<pair<double, int>, double> tmp(input(vec_input[j], vec_input[i], seq_y, co_y, seq_x, co_x, t, u, no_x, no_y));	//TM-align

			align_len = tmp.first.second;	
			Palen.push_back(align_len);	//aligned len
			distance_matrix[i][j] = 1 - tmp.first.first;	//TM-score
			rmsd_matrix[i][j] = tmp.second;	//rmsd
			if (j == 0)
			{
				if (i == 1)
				{
					Iname.push_back(vec_input[0]);	
					Ilen.push_back(len_x);
					vector<char> seq_xv(seq_x, seq_x+len_x);	//save sequence of structure x in vector
					Iseq.push_back(seq_xv);
					for (unsigned int k=0; k<len_x; k++)
					{
						co_xv[k][0] = co_x[k][0];
						co_xv[k][1] = co_x[k][1];
						co_xv[k][2] = co_x[k][2];
					}
					Ico.push_back(co_xv);
					vector<int> no_xv(len_x);	//save order of structure x in vector
					for (unsigned int m=0; m<len_x; m++)
					{
						no_xv[m] = m;
					}
					Ename.push_back(vec_input[0]);
					Elen.push_back(len_x);
					Eno.push_back(no_xv);
				}
				Iname.push_back(vec_input[i]);
				Ilen.push_back(len_y);
				vector<char> seq_yv(seq_y, seq_y+len_y);	//save sequence of structure y in vector 
				Iseq.push_back(seq_yv);
				for (unsigned int k=0; k<len_y; k++)
				{
					co_yv[k][0] = co_y[k][0];
					co_yv[k][1] = co_y[k][1];
					co_yv[k][2] = co_y[k][2];
				}
				Ico.push_back(co_yv);
				vector<int> no_yv(len_y);	//save order of structure y in vector
				for (unsigned int m=0; m<len_y; m++)
				{
					no_yv[m] = m;
				}
				Ename.push_back(vec_input[i]);
				Elen.push_back(len_y);
				Eno.push_back(no_yv);
			}
			vector<vector<double> > u_v(3, vector<double>(3));	//save rotation matrix in vector
			vector<double> t_v(3);	//save shift vector t in vector
			for (unsigned int k=0; k<3; k++)
			{
				for (unsigned int l=0; l<3; l++)
				{
					u_v[k][l] = u[k][l];
				}
			}
			for (unsigned int k=0; k<3; k++)
			{
				t_v[k] = t[k];
			}
			Pt.push_back(t_v);
			Pu.push_back(u_v);
			pair<string, string> Pn(vec_input[i],vec_input[j]);
			Pname.push_back(Pn);
			vector<int> no_xv(no_x, no_x+align_len);	//save order of structure x in vector
			vector<int> no_yv(no_y, no_y+align_len);	//save order of structure y in vector
			Pno1.push_back(no_yv);
			Pno2.push_back(no_xv);
			int loc_x = get_Initial_location(vec_input[i]);
			int loc_y = get_Initial_location(vec_input[j]);
			vector<vector<double> > tmp1(caculate_dis_matrix(Ilen[loc_x], Ilen[loc_y], u_v, t_v, Ico[loc_x], Ico[loc_y]));
			Pdis.push_back(tmp1);
			Pd0.push_back(d0_0);

			delete []seq_x;
			delete []seq_y;
			delete []no_x;
			delete []no_y;
			
			DeleteArray(&co_x, len_x);
			DeleteArray(&co_y, len_y);
			
		}
	}
	fprintf(fp, "%-20s"," ");
	fprintf(fp1, "%-20s"," ");
	fprintf(fp2, "%d",tot_num);
	osfile << tot_num << endl;	//number of input structure
	for (unsigned int i = 0; i<tot_num; i++)	//output matrix.txt
	{
		fprintf(fp, "%-20s", vec_input[i]);
		fprintf(fp1, "%-20s", vec_input[i]);
	}
	fprintf(fp, "\n");
	fprintf(fp1, "\n");
	fprintf(fp2, "\n");
	for (unsigned int i = 0; i<tot_num; i++)	//output matrix.txt
	{
		osfile << vec_input[i] << " ";
		fprintf(fp, "%-20s", vec_input[i]);
		fprintf(fp1, "%-20s", vec_input[i]);
		fprintf(fp2, "%-.10s", vec_input[i]);
		for (unsigned int j = 0; j<tot_num; j++)
		{
			osfile << distance_matrix[i][j] << " ";
			if (j < i)
			{
				fprintf(fp, "%-20.4f", (1 - distance_matrix[i][j]));
				fprintf(fp1, "%-20.3f", rmsd_matrix[i][j]);
				fprintf(fp2, "%-20.4f", distance_matrix[i][j]);
			}
			if (j >= i)
			{
				fprintf(fp, "%-20.4f", (1 - distance_matrix[j][i]));
				fprintf(fp1, "%-20.3f", rmsd_matrix[j][i]);
				fprintf(fp2, "%-20.4f", distance_matrix[j][i]);
			}
			if (distance_matrix[i][j] != 0)
				vec_TM.push_back(1 - distance_matrix[i][j]);
		}
		osfile << endl;
		fprintf(fp, "\n");
		fprintf(fp1, "\n");
		fprintf(fp2, "\n");
	}
	osfile.close();
	fclose(fp);
	fclose(fp1);
	fclose(fp2);
	DeleteArray(&distance_matrix, tot_num);
}

vector<string> //cut string str_in by space 
TMM::segmentation(string str_in)	//string input
{
	vector<string> str_vec;
	int j = 0;
	for (unsigned int i = 0; i<str_in.length(); i++)
	{
		if (str_in[i] == '+')
		{
			string tmp;
			tmp = str_in.substr(j, i-j);
			j = i + 1;
			str_vec.push_back(tmp);
		}
		if (i == str_in.length()-1)
		{
			string tmp;
			tmp = str_in.substr(j, i-j+1);
			str_vec.push_back(tmp);
		}
	}
	return str_vec;
}
	
double	//get score from pairwise score matrix
TMM::get_score_from_matrix(int loc,	//input location 
						   int row,	//input row
						   int col)	//input column
{
	return Pscore[loc][row][col];
}

int //get Element location
TMM::get_Element_location(string name_E)	//input Element name	
{
	int loc = -1;	//location of name_E in vector Ename
	int Ename_siz = Ename.size();	//size of Ename
	for (size_t i=0; i<Ename_siz; i++)
	{
		if(Ename[i] == name_E)
		{
			loc = i;
			break;
		}
	}
	return loc;
}

int //get Inaial location
TMM::get_Initial_location(string name_I)	//input Initial name
{
	int loc = -1;	//location of name_I in vector Iname
	for (size_t i=0; i<tot_num; i++)
	{
		if(Iname[i] == name_I)
		{
			loc = i;
			break;
		}
	}
	return loc;
}

int //get Pair location
TMM::get_Pair_location(string name_P1,string name_P2)	//input pair name1 and name2
{
	int loc = -1;	//location of pairs of name_P1 and name_P2 in vector Pname
	size_t Pname_siz = Pname.size();
	for (size_t i=0; i<Pname.size(); i++)
	{
		if(Pname[i].first == name_P1 && Pname[i].second == name_P2)
		{
			loc = i;
			break;
		}
	}
	return loc;
}

double	//caculate distance square
TMM::caculate_distance_square(vector<double> co_a, //coordinate of a resdiue on structure x
							vector<double> co_b)	//coordinate of a residue on structure y
{
	double dis;	//distance of two residues
	double tmp1, tmp2, tmp3;
	tmp1 = co_a[0] - co_b[0];
	tmp2 = co_a[1] - co_b[1];
	tmp3 = co_a[2] - co_b[2];

	dis = tmp1*tmp1 + tmp2*tmp2 + tmp3*tmp3;
	return dis;
}

vector<vector<double> >	//get score matrix
TMM::caculate_dis_matrix(int len_x,	//length of structure x
							int len_y,	//length of structure y
							vector<vector<double> > u,	//toration matrix
							vector<double> t,	//swift vector
							vector<vector<double> > co_x,	//coordinate of structure x
							vector<vector<double> > co_y)	//d0 in TM-align
{
	double dis, sco;	//distance, score of two residues
	vector<vector<double> > matrix_dis(len_x, vector<double>(len_y));	//score matrix
	vector<vector<double> > co_new_y(len_y, vector<double>(3));	//coordinate of Initial rotatee

	for (unsigned int i=0; i<len_y; i++)	//caculate coordinate after rotation and swift
	{
		co_new_y[i][0] = t[0] + u[0][0] * co_y[i][0] + u[0][1] * co_y[i][1] + u[0][2] * co_y[i][2];
		co_new_y[i][1] = t[1] + u[1][0] * co_y[i][0] + u[1][1] * co_y[i][1] + u[1][2] * co_y[i][2];
		co_new_y[i][2] = t[2] + u[2][0] * co_y[i][0] + u[2][1] * co_y[i][1] + u[2][2] * co_y[i][2];
	}
	for (unsigned int i=0; i<len_x; i++) //caculate distance and score
	{
		for (unsigned int j=0; j<len_y; j++)
		{
			matrix_dis[i][j] = caculate_distance_square(co_x[i],co_new_y[j]);
			
		}
	}

	return matrix_dis;
}

vector<vector<double> > 
TMM::caculate_sco_matrix(int len_x, int len_y, vector<vector<double> > matrix_dis, vector<char> seq_x, vector<char> seq_y, double d0, double TMscore)
{
	double d0_2 = d0 * d0;
	vector<vector<double> > matrix_sco(len_x, vector<double>(len_y));		
	if (TMscore < 0.5)
	{
		for(int i=0; i< len_x; i++)
		{
			for(int j=0; j< len_y; j++)
			{
				if(matrix_dis[i][j] < 16)
				{
					matrix_sco[i][j] = d0_2 / (matrix_dis[i][j] + d0_2);	
				}
				else if(matrix_dis[i][j] > 64)
				{
					matrix_sco[i][j] = -0.1;
				}
				else
				{	
					//matrix_sco[i][j] = 0.1 * (pow(2.71828, (4 - sqrt(matrix_dis[i][j]))) - 1) / pow(2, (0.1 * get_Blosum62_1(seq_x[i], seq_y[j])));
					matrix_sco[i][j] = (pow(2.71828, (4 - sqrt(matrix_dis[i][j]))) - 1) / get_Blosum62(seq_x[i], seq_y[j]);
					//cout<<get_Blosum62(seq_x[i], seq_y[j])<<" "<<10 *pow(2, (0.1 * get_Blosum62_1(seq_x[i], seq_y[j])))<<endl;
				}
			}
		}
	}
	else
	{
		for(int i=0; i< len_x; i++)
		{
			for(int j=0; j< len_y; j++)
			{
				if(matrix_dis[i][j] < 16)
				{
					matrix_sco[i][j] = d0_2 / (matrix_dis[i][j] + d0_2);	
				}
				else if(matrix_dis[i][j] > 64)
				{
					matrix_sco[i][j] = -0.1;
				}
				else
				{	
					matrix_sco[i][j] = 0.1 * (pow(2.71828, (4 - sqrt(matrix_dis[i][j])))) - 0.1;
				}
			}
		}
	}
	
	return matrix_sco;
}

double	//get biger one
TMM::max_of_two(double a,	//number a
				double b)	//number b
{
	return (a>b) ? a : b;
}

double //get smaller one
TMM::min_of_two(double a, 	//number a
				double b)	//number b
{
	return (a>b) ? b : a;
}

vector<vector<double> >	//get score matrix of two Element
TMM::get_score_matrix(string str_x,	//input name of Element x
					string str_y)	//input name of Element y
{
	vector<string> vec_x;	//structure name in Element x
	vector<string> vec_y;	//structure name in Element y 
	vec_x = segmentation(str_x);
	vec_y = segmentation(str_y);
	int loc_x = get_Element_location(str_x);
	int loc_y = get_Element_location(str_y);
	size_t len_x = Elen[loc_x];
	size_t len_y = Elen[loc_y];

	int num_x = vec_x.size();	//number of structure in Element x
	int num_y = vec_y.size();	//number of structure in Element y

	int mul_num = num_x * num_y;
	vector<vector<double> > matrix_sco(len_x, vector<double>(len_y));	//score matrix of Element x and y
	for (size_t i=0; i<len_x; i++)
	{
		for (size_t j=0; j<len_y; j++)
		{
			matrix_sco[i][j] = 0;
		}
	}

	vector<int> vec_loc_a(num_x);	//location of structure in vec_x 
	vector<int> vec_loc_b(num_y);	//location of structure in vec_y
	int loc_a, loc_b;	//location of element in Elenment x and y
	
	for(int i=0; i<num_x; i++)
	{
		vec_loc_a[i] = get_Element_location(vec_x[i]);
	}

	for(int i=0; i<num_y; i++)
	{
		vec_loc_b[i] = get_Element_location(vec_y[i]);
	}
	
	vector<int> max(mul_num);
	vector<int> min(mul_num);

	for(unsigned int i=0; i< num_x; i++)
	{
		for(unsigned int j=0; j< num_y; j++)
		{
			max[i*num_y+j] = max_of_two(vec_loc_a[i], vec_loc_b[j]);
			min[i*num_y+j] = min_of_two(vec_loc_a[i], vec_loc_b[j]);
		}
	}

	for (size_t i=0; i<len_x; i++)
	{
		for (size_t j=0; j<len_y; j++)
		{	
			for (unsigned int l=0; l<num_x; l++)
			{
				for (unsigned int m=0; m<num_y; m++)
				{
					double sco = 0;			
					if (Eno[vec_loc_a[l]][i] != -1 && Eno[vec_loc_b[m]][j] != -1) //no blank
					{
						int iy = l * num_y + m;
						int loc_p = get_Pair_location(Iname[max[iy]], Iname[min[iy]]);
						if(vec_loc_a[l] == max[iy])	
							sco = get_score_from_matrix(loc_p, Eno[vec_loc_a[l]][i], Eno[vec_loc_b[m]][j]);
						if(vec_loc_a[l] == min[iy])
							sco = get_score_from_matrix(loc_p, Eno[vec_loc_b[m]][j], Eno[vec_loc_a[l]][i]);
					}
					if (sco != 0)
						matrix_sco[i][j] += sco;	//summeraise
				}
			}
			//matrix_sco[i][j] = matrix_sco[i][j] / mul_num;	//average		
		}
	}
	return matrix_sco;
}

vector<int>	//dynamic programming
TMM::NWDP_Nor(int len_1,
			int len_2, 
			vector<vector<double> > score_matrix)	//score matrix
{
	vector<int> j2i(len_2);	//align relation of x and y

	int i, j;
	double h, v, d;
	
	vector<vector<double> > vall(len_1+1, vector<double> (len_2+1));
	vector<vector<double> > path(len_1+1, vector<double> (len_2+1));

	vall[0][0] = 0;
	for (i = 0; i <= len_1; i++)
	{
		vall[i][0] = 0;
		path[i][0] = false;
	}

	for (j = 0; j <= len_2; j++)
	{
		vall[0][j] = 0;
		path[0][j] = false;
	}

	for(j=0; j< len_2; j++)
	{
		j2i[j] = -1;
	}

	for (i = 1; i <= len_1; i++)
	{
		for (j = 1; j <= len_2; j++)
		{
			d = vall[i - 1][j - 1] + score_matrix[i - 1][j - 1];
			h = vall[i - 1][j];
			if(path[i-1][j])
				h += -0.2;
			
			v = vall[i][j - 1];
			if(path[i][j-1])
				v += -0.2;

			if (d >= h && d >= v)
			{
				path[i][j] = true;
				vall[i][j] = d;
			}
			else
			{
				path[i][j] = false;
				if (v >= h)
					vall[i][j] = v;
				else
					vall[i][j] = h;
			}
		}
	}

	i = len_1;
	j = len_2;
	
	while (i>0 && j>0)
	{
		if (path[i][j])
		{
			j2i[j - 1] = i - 1;
			i--;
			j--;
		}
		else
		{
			h = vall[i - 1][j];
			if(path[i-1][j]) h += -0.2;

			v = vall[i][j - 1];
			if(path[i][j-1]) v += -0.2;
			if (v >= h)
				j--;
			else
				i--;
		}
	}	
	return j2i;
}

pair<vector<vector<int> >, int>	//if alignment complete in TM-align, use it
TMM::get_first_alignment(string str_x, //input Initial x
						string str_y)	//input Initial y
{
	vector<vector<int> > ans1;
	int align_len;	//aligned length of
	int loc_p = get_Pair_location(str_x, str_y);

	align_len = Palen[loc_p];
	ans1.push_back(Pno1[loc_p]);
	ans1.push_back(Pno2[loc_p]);

	pair<vector<vector<int> >, int> ans(ans1, align_len); 
	return ans;
}

void //renewal Element information
TMM::update_E(string str_x, //Element name which merged
			string str_y, //
			vector<vector<int> > vec_no, //new Element order
			vector<int> vec_ind, //location of renewal
			int ali_len)	//aligned length
{
	string new_name_a = str_x + '+' + str_y;	//new Element name
	string new_name_b = str_y + '+' + str_x; 	//new Element name

	Ename.push_back(new_name_a);
	Elen.push_back(ali_len);

	Ename.push_back(new_name_b);
	Elen.push_back(ali_len);
	for(int i=0; i< vec_ind.size(); i++)	//renewal Element order and length
	{
		Elen.erase(Elen.begin() + vec_ind[i]);
		Eno.erase(Eno.begin() + vec_ind[i]);
		Elen.insert(Elen.begin()+vec_ind[i], ali_len);
		Eno.insert(Eno.begin()+vec_ind[i], vec_no[i]);
	}
	max_len = ali_len;
}

pair<vector<vector<int> >, int>	//align alignment, return new order of input Element and aligned length
TMM::regular_number(string str_x, //input Element x
					string str_y,	//input Element y
					vector<int> ind_x,	
					vector<int> ind_y,	
					vector<vector<int> > no_x,	
					vector<vector<int> > no_y,	
					vector<vector<double> > sco)	//score matrix
{	
	int loc_x = get_Element_location(str_x);	//location of Element x
	int loc_y = get_Element_location(str_y);	//location of Element y

	int len_x = Elen[loc_x]; //length of Element x
	int len_y = Elen[loc_y]; //length of Elemenr y
	int num_x = ind_x.size();	//total number of structure in Element x
	int num_y = ind_y.size(); 	//total number of structure in Element y
	int i, j;	//tmp parameter

	int aligned_len = len_x + len_y;	//max aligned length
	vector<vector<int> > p1(num_x, vector<int>(aligned_len));	//order of Element x
	vector<vector<int> > p2(num_y, vector<int>(aligned_len));	//order of Element y
		//input score matrix

	int k;
	int n_ali = 0;

	vector<int> m1;	//position of aligned for Element x
	vector<int> m2;	//position of aligned for Element y
	
	vector<int> j2i(NWDP_Nor(len_x, len_y, sco));
	
	for (int j=0; j<len_y; j++)
	{
		i = j2i[j];
		if (i >= 0)
		{
			n_ali++;
			m1.push_back(i);
			m2.push_back(j);
		}
	}

	int kk = 0, i_old = 0, j_old = 0;
	for (k=0; k<n_ali; k++)
	{
		for (i=i_old; i< m1[k]; i++)
		{
			for (int m = 0; m < num_x; m++)
			{
				p1[m][kk] = no_x[m][i];
			}
			for (int m = 0; m < num_y; m++)
			{
				p2[m][kk] = -1;
			}
			kk++;
		}

		for (j=j_old; j<m2[k]; j++)
		{
			for (int m=0; m<num_x; m++)
			{
				p1[m][kk] = -1;
			}
			for (int m=0; m<num_y; m++)
			{
				p2[m][kk] = no_y[m][j];
			}
			kk++;
		}

		for (int m=0; m<num_x; m++)
		{
			p1[m][kk] = no_x[m][m1[k]];
		}
		for (int m=0; m<num_y; m++)
		{
			p2[m][kk] = no_y[m][m2[k]];
		}
		kk++;
		i_old = m1[k] + 1;
		j_old = m2[k] + 1;
	}
	
	for (i=i_old; i<len_x; i++)
	{
		for (int m=0; m<num_x; m++)
		{
			p1[m][kk] = no_x[m][i];
		}
		for (int m=0; m<num_y; m++)
		{
			p2[m][kk] = -1;
		}
		kk++;
	}
	
	for (j = j_old; j < len_y; j++)
	{
		for (int m = 0; m < num_x; m++)
		{
			p1[m][kk] = -1;
		}
		for (int m = 0; m < num_y; m++)
		{
			p2[m][kk] = no_y[m][j];
		}
		kk++;
	}	
	for(int i=aligned_len-1; i>= kk; i--)
	{
		for(int j=0; j< num_x; j++)
		{
			p1[j].pop_back();	
		}
		
		for(int j=0; j< num_y; j++)
		{
			p2[j].pop_back();	
		}
	}

	vector<vector<int> > ans1;
	for(int i=0; i< num_x; i++)
	{
		ans1.push_back(p1[i]);
	}
	for(int i=0; i< num_y; i++)
	{
		ans1.push_back(p2[i]);
	}

	pair<vector<vector<int> >, int> ans2(ans1, kk);
	return ans2;
}

void	//save result to seq.txt
TMM::save_seq()
{
	ofstream osfile;
	osfile.open("result.fasta");
	//osfile << tot_num << endl;
	for (unsigned int i = 0; i < tot_num; i++)
	{
		string name1=Iname[i];
		osfile << '>' << name1 << '\n';	//structure name
		int k = 0;
		for (int j = 0; j < Elen[0]; j++)
		{
			
			if (Eno[i][j] != -1)
			{
				osfile << Iseq[i][Eno[i][j]];
			}
			else
			{
				if (Eno[i][j] == -1)
				{
					osfile << '-';
				}
			}
			k ++;
			if (k % 60 == 0)
				osfile<<endl;
		}
		
		osfile << endl;
	}
	osfile.close();
}


bool	//determine whether string a belongs to initial structure name 
TMM::exist_in_initial(string a)
{
	bool ans = false;
	for (int i = 0; i <Iname.size(); i++)
	{
		if(a == Iname[i])
		{
			ans = true;
			break;
		}
	}
	return ans;
}

vector<int>
TMM::Align_ini(pair<string, string> aa)
{
	pair<vector<vector<int> >, int> tmp1;	//result of alignment, consisted of new order and aligned length
	vector<int> tmp2;	//loaction of update
	vector<vector<int> > tmp3;	//new order used in update
	int ali_len;
	int loc_x = get_Initial_location(aa.first);
	int loc_y = get_Initial_location(aa.second);
	int max_loc = max_of_two(loc_x, loc_y);
	int min_loc = min_of_two(loc_x, loc_y);
	tmp1 = get_first_alignment(Iname[max_loc], Iname[min_loc]);

	tmp2.push_back(max_loc);
	tmp2.push_back(min_loc);
		
	tmp3 = tmp1.first;
	ali_len = tmp1.second;
	update_E(Iname[max_loc], Iname[min_loc], tmp3, tmp2, ali_len);
	return tmp2;
}

vector<int>
TMM::Align_unini(pair<string, string> aa, vector<vector<double> > sco)
{
	pair<vector<vector<int> >, int> tmp1;	//result of alignment, consisted of new order and aligned length
	vector<int> tmp2;	//loaction of update
	vector<vector<int> > tmp3;	//new order used in update
	int ali_len;
	vector<string> vec_x(segmentation(aa.first));	//structure name in Element x
	vector<string> vec_y(segmentation(aa.second));	//structure name in Element y

	int num_x = vec_x.size();	//total number of structure in Element x
	int num_y = vec_y.size(); 	//total number of structure in Element y
	vector<int> vec_loc_x(num_x);	//location of structure in Element x
	vector<int> vec_loc_y(num_y);	//location of structure in Element y

	for(int i=0; i<num_x; i++)
	{
		vec_loc_x[i] = get_Element_location(vec_x[i]);
	}
	for(int i=0; i<num_y; i++)
	{
		vec_loc_y[i] = get_Element_location(vec_y[i]);
	}
	vector<vector<int> > no_x;	
	vector<vector<int> > no_y;
	for(int i=0; i< num_x; i++)
	{
		no_x.push_back(Eno[vec_loc_x[i]]);
	}
	for(int i=0; i< num_y; i++)
	{
		no_y.push_back(Eno[vec_loc_y[i]]);
	}
			
	tmp1 = regular_number(aa.first, aa.second, vec_loc_x, vec_loc_y, no_x, no_y, sco);	//align alignment
	for(int i=0; i< num_x; i++)
	{
		tmp2.push_back(vec_loc_x[i]);
	}
	for(int i=0; i< num_y; i++)
	{
		tmp2.push_back(vec_loc_y[i]);
	}
	tmp3 = tmp1.first;
	ali_len = tmp1.second;
	update_E(aa.first, aa.second, tmp3, tmp2, ali_len);
	return tmp2;
}

vector<int>	//align Element whose name saved in pair
TMM::Align_one_step(pair<string, string> aa, vector<vector<double> > sco)	//two name of Element aligned
{
	vector<int> ans;
	if(exist_in_initial(aa.first) == true && exist_in_initial(aa.second) == true)	//use TM-align result
	{
		ans = Align_ini(aa);
	}
	if(exist_in_initial(aa.first) == false || exist_in_initial(aa.second) == false)
	{
		ans = Align_unini(aa, sco);	
	}
	return ans;
}

void //align follow vec_ord
TMM::Align(vector<pair<string, string> > vec_ord)
{
	for (int i = 0; i <tot_num-1; i++)
	{
		//cout<<vec_ord[i].first<<" "<<vec_ord[i].second<<endl;
		vector<vector<double> > sco(get_score_matrix(vec_ord[i].first, vec_ord[i].second));	//score matrix of two Element
		Align_one_step(vec_ord[i], sco);
	}
}
	
double
TMM::get_Blosum62(char a, char b)
{
	double ans = -100;
	if(a == 'C')
	{
		if(b == 'C') ans = 18.6607;     else if(b == 'S') ans = 9.3303; else if(b == 'T') ans = 9.3303; else if(b == 'P') ans = 8.1225; else if(b == 'A') ans = 10;
		else if(b == 'G') ans = 8.1225; else if(b == 'N') ans = 8.1225; else if(b == 'D') ans = 8.1225; else if(b == 'E') ans = 7.5786; else if(b == 'Q') ans = 8.1225;
		else if(b == 'H') ans = 8.1225; else if(b == 'R') ans = 8.1225; else if(b == 'K') ans = 8.1225; else if(b == 'M') ans = 9.3303; else if(b == 'I') ans = 9.3303;
		else if(b == 'L') ans = 9.3303; else if(b == 'V') ans = 9.3303; else if(b == 'F') ans = 8.7055; else if(b == 'Y') ans = 8.7055; else if(b == 'W') ans = 8.7055;
	}
	else if(a == 'S')
	{
		if(b == 'C') ans = 9.3303;      else if(b == 'S') ans = 13.1951; else if(b == 'T') ans = 10.7177; else if(b == 'P') ans = 9.3303; else if(b == 'A') ans = 10.7177;
		else if(b == 'G') ans = 10;	    else if(b == 'N') ans = 10.7177; else if(b == 'D') ans = 10;      else if(b == 'E') ans = 10; 	  else if(b == 'Q') ans = 10;
		else if(b == 'H') ans = 9.3303; else if(b == 'R') ans = 9.3303;  else if(b == 'K') ans = 10;      else if(b == 'M') ans = 9.3303; else if(b == 'I') ans = 8.7055;
		else if(b == 'L') ans = 8.7055; else if(b == 'V') ans = 8.7055;  else if(b == 'F') ans = 8.7055;  else if(b == 'Y') ans = 8.7055; else if(b == 'W') ans = 8.1225;
	}
	else if(a == 'T')
	{
		if(b == 'C') ans = 9.3303;      else if(b == 'S') ans = 10.7177; else if(b == 'T') ans = 13.1951;  else if(b == 'P') ans = 10.7177; else if(b == 'A') ans = 9.3303;
		else if(b == 'G') ans = 10.7177;else if(b == 'N') ans = 10;	     else if(b == 'D') ans = 10.7177;  else if(b == 'E') ans = 10;	    else if(b == 'Q') ans = 10;
		else if(b == 'H') ans = 10;	    else if(b == 'R') ans = 9.3303;  else if(b == 'K') ans = 10;	   else if(b == 'M') ans = 9.3303;  else if(b == 'I') ans = 8.7055;
		else if(b == 'L') ans = 8.7055; else if(b == 'V') ans = 8.7055;  else if(b == 'F') ans = 8.7055;   else if(b == 'Y') ans = 8.7055;  else if(b == 'W') ans = 8.1225;
	}
	else if(a == 'P')
	{
		if(b == 'C') ans = 8.1225;      else if(b == 'S') ans = 9.3303;  else if(b == 'T') ans = 10.7177; else if(b == 'P') ans = 16.2450; else if(b == 'A') ans = 9.3303;
		else if(b == 'G') ans = 8.7055; else if(b == 'N') ans = 9.3303;  else if(b == 'D') ans = 9.3303;  else if(b == 'E') ans = 9.3303;  else if(b == 'Q') ans = 9.3303;
		else if(b == 'H') ans = 8.7055; else if(b == 'R') ans = 8.7055;  else if(b == 'K') ans = 9.3303;  else if(b == 'M') ans = 8.7055;  else if(b == 'I') ans = 8.1225;
		else if(b == 'L') ans = 8.1225; else if(b == 'V') ans = 8.7055;  else if(b == 'F') ans = 7.5786;  else if(b == 'Y') ans = 8.1225;  else if(b == 'W') ans = 7.5786;
	}
	else if(a == 'A')
	{
		if(b == 'C') ans = 10;          else if(b == 'S') ans = 10.7177; else if(b == 'T') ans = 9.3303; else if(b == 'P') ans = 9.3303; else if(b == 'A') ans = 13.1951;
		else if(b == 'G') ans = 10;     else if(b == 'N') ans = 8.7055;  else if(b == 'D') ans = 8.7055; else if(b == 'E') ans = 9.3303; else if(b == 'Q') ans = 9.3303;
		else if(b == 'H') ans = 8.7055; else if(b == 'R') ans = 9.3303;  else if(b == 'K') ans = 9.3303; else if(b == 'M') ans = 9.3303; else if(b == 'I') ans = 9.3303;
		else if(b == 'L') ans = 9.3303; else if(b == 'V') ans = 10;      else if(b == 'F') ans = 8.7055; else if(b == 'Y') ans = 8.7055; else if(b == 'W') ans = 8.1225;
	}
	else if(a == 'G')
	{
		if(b == 'C') ans = 8.1225;       else if(b == 'S') ans = 10;     else if(b == 'T') ans = 10.7177; else if(b == 'P') ans = 8.7055; else if(b == 'A') ans = 10;
		else if(b == 'G') ans = 15.1572; else if(b == 'N') ans = 10;     else if(b == 'D') ans = 9.3303;  else if(b == 'E') ans = 8.7055; else if(b == 'Q') ans = 8.7055;
		else if(b == 'H') ans = 8.7055;  else if(b == 'R') ans = 8.7055; else if(b == 'K') ans = 8.7055;  else if(b == 'M') ans = 8.1225; else if(b == 'I') ans = 7.5786;
		else if(b == 'L') ans = 7.5786;  else if(b == 'V') ans = 8.1225; else if(b == 'F') ans = 8.1225;  else if(b == 'Y') ans = 8.1225; else if(b == 'W') ans = 8.7055;
	}
	else if(a == 'N')
	{
		if(b == 'C') ans = 8.1225;       else if(b == 'S') ans = 10.7177; else if(b == 'T') ans = 10;      else if(b == 'P') ans = 8.7055; else if(b == 'A') ans = 8.7055;
		else if(b == 'G') ans = 10;      else if(b == 'N') ans = 15.1572; else if(b == 'D') ans = 10.7177; else if(b == 'E') ans = 10;     else if(b == 'Q') ans = 10;
		else if(b == 'H') ans = 10.7177; else if(b == 'R') ans = 10;      else if(b == 'K') ans = 10;      else if(b == 'M') ans = 8.7055; else if(b == 'I') ans = 8.1225;
		else if(b == 'L') ans = 8.1225;  else if(b == 'V') ans = 8.1225;  else if(b == 'F') ans = 8.1225;  else if(b == 'Y') ans = 8.7055; else if(b == 'W') ans = 7.5786;
	}
	else if(a == 'D')
	{
		if(b == 'C') ans = 8.1225;      else if(b == 'S') ans = 10;      else if(b == 'T') ans = 10.7177; else if(b == 'P') ans = 9.3303;  else if(b == 'A') ans = 8.7055;
		else if(b == 'G') ans = 9.3303; else if(b == 'N') ans = 10.7177; else if(b == 'D') ans = 15.1572; else if(b == 'E') ans = 11.4870; else if(b == 'Q') ans = 10;
		else if(b == 'H') ans = 9.3303; else if(b == 'R') ans = 8.7055;  else if(b == 'K') ans = 9.3303;  else if(b == 'M') ans = 8.1225;  else if(b == 'I') ans = 8.1225;
		else if(b == 'L') ans = 7.5786; else if(b == 'V') ans = 8.1225;  else if(b == 'F') ans = 8.1225;  else if(b == 'Y') ans = 8.1225;  else if(b == 'W') ans = 7.5786;
	}
	else if(a == 'E')
	{
		if(b == 'C') ans = 7.5786;      else if(b == 'S') ans = 10;     else if(b == 'T') ans = 10;      else if(b == 'P') ans = 9.3303;  else if(b == 'A') ans = 9.3303;
		else if(b == 'G') ans = 8.7055; else if(b == 'N') ans = 10;     else if(b == 'D') ans = 11.4870; else if(b == 'E') ans = 14.1421; else if(b == 'Q') ans = 11.4870;
		else if(b == 'H') ans = 10;     else if(b == 'R') ans = 10;     else if(b == 'K') ans = 10.7177; else if(b == 'M') ans = 8.7055;  else if(b == 'I') ans = 8.1225;
		else if(b == 'L') ans = 8.1225; else if(b == 'V') ans = 8.7055; else if(b == 'F') ans = 8.1225;  else if(b == 'Y') ans = 8.7055;  else if(b == 'W') ans = 8.1225;
	}
	else if(a == 'Q')
	{
		if(b == 'C') ans = 8.1225;      else if(b == 'S') ans = 10;      else if(b == 'T') ans = 10;      else if(b == 'P') ans = 9.3303;  else if(b == 'A') ans = 9.3303;
		else if(b == 'G') ans = 8.7055; else if(b == 'N') ans = 10;      else if(b == 'D') ans = 10;      else if(b == 'E') ans = 11.4870; else if(b == 'Q') ans = 14.1421;
		else if(b == 'H') ans = 10;     else if(b == 'R') ans = 10.7177; else if(b == 'K') ans = 10.7177; else if(b == 'M') ans = 10;      else if(b == 'I') ans = 8.1225;
		else if(b == 'L') ans = 8.7055; else if(b == 'V') ans = 8.7055;  else if(b == 'F') ans = 8.1225;  else if(b == 'Y') ans = 9.3303;  else if(b == 'W') ans = 8.7055;
	}
	else if(a == 'H')
	{
		if(b == 'C') ans = 8.1225;       else if(b == 'S') ans = 9.3303;  else if(b == 'T') ans = 10;     else if(b == 'P') ans = 8.7055;  else if(b == 'A') ans = 8.7055;
		else if(b == 'G') ans = 8.7055;  else if(b == 'N') ans = 10.7177; else if(b == 'D') ans = 9.3303; else if(b == 'E') ans = 10;      else if(b == 'Q') ans = 10;
		else if(b == 'H') ans = 17.4110; else if(b == 'R') ans = 10;      else if(b == 'K') ans = 9.3303; else if(b == 'M') ans = 8.7055;  else if(b == 'I') ans = 8.1225;
		else if(b == 'L') ans = 8.1225;  else if(b == 'V') ans = 8.1225;  else if(b == 'F') ans = 9.3303; else if(b == 'Y') ans = 11.4870; else if(b == 'W') ans = 8.7055;
	}
	else if(a == 'R')
	{
		if(b == 'C') ans = 8.1225;      else if(b == 'S') ans = 9.3303;  else if(b == 'T') ans = 9.3303;  else if(b == 'P') ans = 8.7055; else if(b == 'A') ans = 9.3303;
		else if(b == 'G') ans = 8.7055; else if(b == 'N') ans = 10;      else if(b == 'D') ans = 8.7055;  else if(b == 'E') ans = 10;     else if(b == 'Q') ans = 10.7177;
		else if(b == 'H') ans = 10;     else if(b == 'R') ans = 14.1421; else if(b == 'K') ans = 11.4870; else if(b == 'M') ans = 9.3303; else if(b == 'I') ans = 8.1225;
		else if(b == 'L') ans = 8.7055; else if(b == 'V') ans = 8.1225;  else if(b == 'F') ans = 8.1225;  else if(b == 'Y') ans = 8.7055; else if(b == 'W') ans = 8.1225;
	}
	else if(a == 'K')
	{
		if(b == 'C') ans = 8.1225;      else if(b == 'S') ans = 10;      else if(b == 'T') ans = 10;      else if(b == 'P') ans = 9.3303;  else if(b == 'A') ans = 9.3303;
		else if(b == 'G') ans = 8.7055; else if(b == 'N') ans = 10;      else if(b == 'D') ans = 9.3303;  else if(b == 'E') ans = 10.7177; else if(b == 'Q') ans = 10.7177;
		else if(b == 'H') ans = 9.3303; else if(b == 'R') ans = 11.4870; else if(b == 'K') ans = 14.1421; else if(b == 'M') ans = 9.3303;  else if(b == 'I') ans = 8.1225;
		else if(b == 'L') ans = 8.7055; else if(b == 'V') ans = 8.7055;  else if(b == 'F') ans = 8.1225;  else if(b == 'Y') ans = 8.7055;  else if(b == 'W') ans = 8.1225;
	}
	else if(a == 'M')
	{
		if(b == 'C') ans = 9.3303;       else if(b == 'S') ans = 9.3303;  else if(b == 'T') ans = 9.3303; else if(b == 'P') ans = 8.7055;  else if(b == 'A') ans = 9.3303;
		else if(b == 'G') ans = 8.1225;  else if(b == 'N') ans = 8.7055;  else if(b == 'D') ans = 8.1225; else if(b == 'E') ans = 8.7055;  else if(b == 'Q') ans = 10;
		else if(b == 'H') ans = 8.7055;  else if(b == 'R') ans = 9.3303;  else if(b == 'K') ans = 9.3303; else if(b == 'M') ans = 14.1421; else if(b == 'I') ans = 10.7177;
		else if(b == 'L') ans = 11.4870; else if(b == 'V') ans = 10.7177; else if(b == 'F') ans = 10;     else if(b == 'Y') ans = 9.3303;  else if(b == 'W') ans = 9.3303;
	}
	else if(a == 'I')
	{
		if(b == 'C') ans = 9.3303; 		 else if(b == 'S') ans = 8.7055;  else if(b == 'T') ans = 8.7055; else if(b == 'P') ans = 8.1225;  else if(b == 'A') ans = 9.3303;
		else if(b == 'G') ans = 7.5786;  else if(b == 'N') ans = 8.1225;  else if(b == 'D') ans = 8.1225; else if(b == 'E') ans = 8.1225;  else if(b == 'Q') ans = 8.1225;
		else if(b == 'H') ans = 8.1225;  else if(b == 'R') ans = 8.1225;  else if(b == 'K') ans = 8.1225; else if(b == 'M') ans = 10.7177; else if(b == 'I') ans = 13.1951;
		else if(b == 'L') ans = 11.4870; else if(b == 'V') ans = 12.3114; else if(b == 'F') ans = 10; 	  else if(b == 'Y') ans = 9.3303;  else if(b == 'W') ans = 8.1225;
	}
	else if(a == 'L')
	{
		if(b == 'C') ans = 9.3303; 		 else if(b == 'S') ans = 8.7055;  else if(b == 'T') ans = 8.7055; else if(b == 'P') ans = 8.1225;  else if(b == 'A') ans = 9.3303;
		else if(b == 'G') ans = 7.5786;  else if(b == 'N') ans = 8.1225;  else if(b == 'D') ans = 7.5786; else if(b == 'E') ans = 8.1225;  else if(b == 'Q') ans = 8.7055;
		else if(b == 'H') ans = 8.1225;  else if(b == 'R') ans = 8.7055;  else if(b == 'K') ans = 8.7055; else if(b == 'M') ans = 11.4870; else if(b == 'I') ans = 11.4870;
		else if(b == 'L') ans = 13.1951; else if(b == 'V') ans = 10.7177; else if(b == 'F') ans = 10; 	  else if(b == 'Y') ans = 9.3303;  else if(b == 'W') ans = 8.7055;
	}
	else if(a == 'V')
	{
		if(b == 'C') ans = 9.3303; 		 else if(b == 'S') ans = 8.7055;  else if(b == 'T') ans = 8.7055; else if(b == 'P') ans = 8.7055;  else if(b == 'A') ans = 10;
		else if(b == 'G') ans = 8.1225;  else if(b == 'N') ans = 8.1225;  else if(b == 'D') ans = 8.1225; else if(b == 'E') ans = 8.7055;  else if(b == 'Q') ans = 8.7055;
		else if(b == 'H') ans = 8.1225;  else if(b == 'R') ans = 8.1225;  else if(b == 'K') ans = 8.7055; else if(b == 'M') ans = 10.7177; else if(b == 'I') ans = 12.3114;
		else if(b == 'L') ans = 10.7177; else if(b == 'V') ans = 13.1951; else if(b == 'F') ans = 9.3303; else if(b == 'Y') ans = 9.3303;  else if(b == 'W') ans = 8.1225;
	}
	else if(a == 'F')
	{
		if(b == 'C') ans = 8.7055; 		else if(b == 'S') ans = 8.7055; else if(b == 'T') ans = 8.7055;  else if(b == 'P') ans = 7.5786;  else if(b == 'A') ans = 8.7055;
		else if(b == 'G') ans = 8.1225; else if(b == 'N') ans = 8.1225; else if(b == 'D') ans = 8.1225;  else if(b == 'E') ans = 8.1225;  else if(b == 'Q') ans = 8.1225;
		else if(b == 'H') ans = 9.3303; else if(b == 'R') ans = 8.1225; else if(b == 'K') ans = 8.1225;  else if(b == 'M') ans = 10;	  else if(b == 'I') ans = 10;
		else if(b == 'L') ans = 10; 	else if(b == 'V') ans = 9.3303; else if(b == 'F') ans = 15.1572; else if(b == 'Y') ans = 12.3114; else if(b == 'W') ans = 10.7177;
	}
	else if(a == 'Y')
	{
		if(b == 'C') ans = 8.7055;		 else if(b == 'S') ans = 8.7055; else if(b == 'T') ans = 8.7055;  else if(b == 'P') ans = 8.1225;  else if(b == 'A') ans = 8.7055;
		else if(b == 'G') ans = 8.1225;  else if(b == 'N') ans = 8.7055; else if(b == 'D') ans = 8.1225;  else if(b == 'E') ans = 8.7055;  else if(b == 'Q') ans = 9.3303;
		else if(b == 'H') ans = 11.4870; else if(b == 'R') ans = 8.7055; else if(b == 'K') ans = 8.7055;  else if(b == 'M') ans = 9.3303;  else if(b == 'I') ans = 9.3303;
		else if(b == 'L') ans = 9.3303;  else if(b == 'V') ans = 9.3303; else if(b == 'F') ans = 12.3114; else if(b == 'Y') ans = 16.2450; else if(b == 'W') ans = 11.4870;
	}
	else if(a == 'W')
	{
		if(b == 'C') ans = 8.7055; 		else if(b == 'S') ans = 8.1225; else if(b == 'T') ans = 8.1225;  else if(b == 'P') ans = 7.5786;  else if(b == 'A') ans = 8.1225;
		else if(b == 'G') ans = 8.7055; else if(b == 'N') ans = 7.5786; else if(b == 'D') ans = 7.5786;  else if(b == 'E') ans = 8.1225;  else if(b == 'Q') ans = 8.7055;
		else if(b == 'H') ans = 8.7055; else if(b == 'R') ans = 8.1225; else if(b == 'K') ans = 8.1225;  else if(b == 'M') ans = 9.3303;  else if(b == 'I') ans = 8.1225;
		else if(b == 'L') ans = 8.7055; else if(b == 'V') ans = 8.1225; else if(b == 'F') ans = 10.7177; else if(b == 'Y') ans = 11.4870; else if(b == 'W') ans = 21.4355;
	}
	return ans;
}

double 
TMM::get_average_pair_TMscore()
{
	double ans = 0;
	int vec_siz = vec_TM.size();
	for(int i=0; i< vec_siz; i++)
	{
		ans += vec_TM[i];
	}
	ans = ans/vec_siz;
	return ans;
}

void	//to align
TMM::programming(vector<char*> vec_input, char *out)
{		
	if (vec_input.size() == 0)	//no input in input_list
	{
		cout<<"no input structure!"<<endl;

	}
	if (vec_input.size() == 1)	//1 input structure
	{
		cout<<"only one input structure!"<<endl;
	}
	if (vec_input.size() == 2)	//2 input structures, do TM-align
	{
		load_all();
		pair<string, string> in(vec_input[0], vec_input[1]);
		Align_ini(in);
		show_inf();
		save_seq();	//save result in seq.txt

		//seq_to_number("seq.txt");
		get_new_co();
		get_dis_vec();
		//vector<int> loc(get_CC_all(1));
		//vector<int> CC_loc(get_CC(4, loc));
		cout<<endl;

		cout<<"------------- Summary of the MSTA  -------------"<<endl;
		//cout<<endl;
		//cout<<"Common core: "<<CC_loc.size()<<endl;
		//cout<<"CC RMSD: "<<multi_RMSD(CC_loc)<<endl;
		//cout<<"CC TMscore: "<<get_average_TM_score(CC_loc)<<endl;
		cout<<"L_ali: "<<average_pc(4)<<endl;
		cout<<"RMSD: "<<multi_pc_RMSD()<<endl;
		cout<<"TMscore: "<<get_average_TM_score(4)<<endl;
		cout<<endl;
		show_seq();
		output_file(0, out);
		DeleteArray(&ato, tot_num);
		DeleteArray(&res, tot_num);
		DeleteArray(&num1, tot_num);
		DeleteArray(&num2, tot_num);
		DeleteArray(&conect, tot_num);

		for (int i=1; i< tot_num; i++)
		{
			DeleteArray(&full_co[i], full_len[i]);
		}

		delete [] full_len;
		delete [] conect_len;
	}
	if (vec_input.size() > 2)	//do mTM-align
	{
		load_all();
		Tree.read_file("./matrix.txt");	
		Tree.construction_of_tree();	//construct phylogenetic tree
		vector<pair<string,string> > aaa(Tree.find_alignment_target());	//vector of align order
		double ave_TM = get_average_pair_TMscore();
		int k=0;
		for (int i=0; i<tot_num; i++)	//save information
		{
			for (int j=0; j<i; j++)
			{
				double tt[3];
				double uu[3][3];
				for(int m=0; m< 3; m++)
				{
					tt[m] = Pt[k][m];
					for(int l=0; l< 3; l++)
					{
						uu[m][l] = Pu[k][m][l];
					}
				}
				vector<vector<double> > tmp(caculate_sco_matrix(Ilen[i], Ilen[j], Pdis[k], Iseq[i], Iseq[j], Pd0[k], ave_TM));
				output_pair(Iname[i], Iname[j], i, j, tt, uu);
				Pscore.push_back(tmp);
				k++;
			}
		}
		//cout<<get_average_pair_TMscore()<<endl;
		Align(aaa);	//alignshow_inf();
		save_seq();	//save result in seq.txt
		show_inf();
		//seq_to_number("seq.txt");
		get_new_co();
		get_dis_vec();
		vector<int> loc(get_CC_all(1));
		vector<int> CC_loc(get_CC(4, loc));

		cout<<endl;
		cout<<"------------- Summary of the MSTA  -------------"<<endl;
		//cout<<endl;
		cout<<"Lcore: "<<CC_loc.size()<<endl;
		cout<<"ccRMSD: "<<multi_RMSD(CC_loc)<<endl;
		cout<<"ccTM-score: "<<get_average_cc_TM_score(CC_loc)<<endl;
		cout<<"Lali: "<<average_pc(4)<<endl;
		cout<<"RMSD: "<<multi_pc_RMSD()<<endl;
		cout<<"TM-score: "<<get_average_TM_score(4)<<endl;
		cout<<endl;
		show_seq();
		cor = select_core();
		if (tot_num <=61)
			output_file(cor, out);
		else
			output_file_mod(cor, out);
		DeleteArray(&ato, tot_num);
		DeleteArray(&res, tot_num);
		DeleteArray(&num1, tot_num);
		DeleteArray(&num2, tot_num);

		for (int i=1; i< tot_num; i++)
		{
			DeleteArray(&full_co[i], full_len[i]);
		}

		delete [] full_len;
	}
}

void 
TMM::load_all()
{    
    full_len = new int [tot_num];
   for(int i=0; i< tot_num; i++)
    {
    	full_len[i] = get_full_len(Iname[i]);
    }

	full_co = new double **[tot_num];
	for (int i = 0; i < tot_num; i++)
	{
		full_co[i] = new double *[full_len[i]];
	}
	for (int i = 0; i < tot_num;i++)
	{
		for (int j = 0; j < full_len[i]; j++)
		{
			full_co[i][j] = new double[3];
		}
	}
	
	num1 = new char **[tot_num];
	for (int i = 0; i < tot_num; i++)
	{
		num1[i] = new char *[full_len[i]];
	}
	for (int i = 0; i < tot_num;i++)
	{
		for (int j = 0; j < full_len[i]; j++)
		{
			num1[i][j] = new char[5];
		}
	}

	num2 = new char **[tot_num];
	for (int i = 0; i < tot_num; i++)
	{
		num2[i] = new char *[full_len[i]];
	}
	for (int i = 0; i < tot_num;i++)
	{
		for (int j = 0; j < full_len[i]; j++)
		{
			num2[i][j] = new char[5];
		}
	}

	ato = new char **[tot_num];
	for (int i = 0; i < tot_num; i++)
	{
		ato[i] = new char *[full_len[i]];
	}
	for (int i = 0; i < tot_num;i++)
	{
		for (int j = 0; j < full_len[i]; j++)
		{
			ato[i][j] = new char[5];
		}
	}

	fla = new char **[tot_num];
	for (int i = 0; i < tot_num; i++)
	{
		fla[i] = new char *[full_len[i]];
	}
	for (int i = 0; i < tot_num;i++)
	{
		for (int j = 0; j < full_len[i]; j++)
		{
			fla[i][j] = new char[6];
		}
	}
	
	res = new char **[tot_num];
	for (int i = 0; i < tot_num; i++)
	{
		res[i] = new char *[full_len[i]];
	}
	for (int i = 0; i < tot_num;i++)
	{
		for (int j = 0; j < full_len[i]; j++)
		{
			res[i][j] = new char[3];
		}
	}
	
	for (int i = 0; i < tot_num;i++)
	{
		read_PDB_fullatom(Iname[i], fla[i], ato[i], res[i], num1[i], num2[i], full_co[i]);
	}

	conect_len = new int [tot_num];
    for(int i=0; i< tot_num; i++)
    {
    	conect_len[i] = get_CONECT_len(Iname[i]);
    }

	conect = new string *[tot_num];
	for (int i = 0; i < tot_num; i++)
	{
		conect[i] = new string [conect_len[i]];
	}
	
	for (int i=0; i< tot_num; i++)
	{
		read_CONECT(Iname[i], conect[i]);
	}
	//cout<<"complete"<<endl;
}

bool in_or_not(int a, vector<int> vec_a)
{
	for (int i=0; i< vec_a.size(); i++)
	{
		if(vec_a[i] == a)
		{
			return true;
		}
	}
	return false;
}

void 
TMM::output_pair(char *xname,
		char *yname,
		int a,
		int b,
		double t0[3],
		double u0[3][3])
{
	double **coord;
    NewArray(&coord, full_len[b], 3);
    do_rotation(full_co[b], coord, full_len[b], t0, u0);

    std::stringstream ss;
	ss << yname << "+" << xname << "_pair.pdb";
	string out_name_str = ss.str();

	char *out_name = new char[strlen(out_name_str.c_str())+1];
	strcpy(out_name, out_name_str.c_str());
    //cout<<out_name<<endl;
    FILE *fp = fopen(out_name, "w");
	
	fprintf(fp, "REMARK%20s\n",yname);
	for (int i = 0; i < full_len[b]; i++)
	{
		fprintf(fp, "%6s%5s %-4s %3s %1c%5s   %8.3f%8.3f%8.3f\n", fla[b][i], num1[b][i], ato[b][i], res[b][i], 'A', num2[b][i], coord[i][0], coord[i][1], coord[i][2]);
		//cout<<fla[b][i]<<" "<<num1[b][i]<<" "<<ato[b][i]<<" "<<res[b][i]<<" "<<num2[b][i]<<endl;
	}
	for(int i=0; i < conect_len[b]; i++)
	{
		fprintf(fp, "%s\n", conect[b][i].c_str());
	}
	fprintf(fp, "TER\n");
	//fprintf(fp, "%s\n","ENDMDL");
	fprintf(fp, "\n");

	//fprintf(fp, "%s      %d\n","MODEL",loc);
	fprintf(fp, "REMARK%20s\n",xname);
	for (int i = 0; i < full_len[a]; i++)
		fprintf(fp, "%6s%5s %-4s %3s %1c%5s   %8.3f%8.3f%8.3f\n", fla[a][i], num1[a][i], ato[a][i], res[a][i], 'B', num2[a][i], full_co[a][i][0], full_co[a][i][1], full_co[a][i][2]);
	for(int i=0; i < conect_len[a]; i++)
	{
		fprintf(fp, "%s\n", conect[a][i].c_str());
	} 
	fprintf(fp, "TER\n");
	fprintf(fp, "\n");
	//fprintf(fp, "%s\n","ENDMDL");
	fclose(fp);
}

void 
TMM::output_x(int a, int b, int kk, char z, char *out)
{
	double t[3];
	double u[3][3];
	int max = 2000;
	int *m1,*m2;
	double **xtm, **ytm;
    double rmsd;

    double **coord;
    NewArray(&coord, full_len[a], 3);
    m1 = new int[Ilen[a]]; //alignd index in x
    m2 = new int[Ilen[b]]; //alignd index in y
    int k1=0;
    	
    vector<int> CC(get_CC_all(1));

    for(int i=0;i<CC.size();i++)
    {
        m1[k1]=Eno[a][CC[i]];
        m2[k1]=Eno[b][CC[i]];
        k1++;
    }

    NewArray(&xtm, k1, 3);
    NewArray(&ytm, k1, 3);
    for (int j = 0; j<k1; j++)
	{
		xtm[j][0] = Ico[a][m1[j]][0];
		xtm[j][1] = Ico[a][m1[j]][1];
		xtm[j][2] = Ico[a][m1[j]][2];

		ytm[j][0] = Ico[b][m2[j]][0];
		ytm[j][1] = Ico[b][m2[j]][1];
		ytm[j][2] = Ico[b][m2[j]][2];
	}

	Kabsch(xtm, ytm, k1, 1, &rmsd, t, u);
	do_rotation(full_co[a], coord, full_len[a], t, u);

	FILE *fp = fopen(out, "a");
	
	//superposed structure B
	//fprintf(fp, "%s      %d\n","MODEL",a);
	fprintf(fp, "REMARK%20s\n",Iname[a]);
	for (int i = 0; i < full_len[a]; i++)
	{
		fprintf(fp, "%6s%5s %-4s %3s %1c%5s   %8.3f%8.3f%8.3f\n", fla[a][i], num1[a][i], ato[a][i], res[a][i],z, num2[a][i], coord[i][0], coord[i][1], coord[i][2]);
		//cout<<fla[a][i]<<" "<<num1[a][i]<<" "<<ato[a][i]<<" "<<res[a][i]<<" "<<z<<" "<<num2[a][i]<<endl;
	}
	for(int i=0; i < conect_len[a]; i++)
	{
		fprintf(fp, "%s\n", conect[a][i].c_str());
	}
	fprintf(fp, "TER\n");
	//fprintf(fp, "%s\n","ENDMDL");
	fprintf(fp, "\n");
	fclose(fp);

	vector<int> CC_loc(get_CC(4, CC));
	vector< vector<int> > CC_each(tot_num, vector<int>(CC_loc.size()));
	for(int i=0; i<CC_loc.size(); i++)
	{
		for(int j=0; j<tot_num; j++)
		{
			CC_each[j][i] = Eno[j][CC_loc[i]];
		}
	}


	int tmp1 = atoi(num2[a][0]);
	int count = 0;
	FILE *fp1 = fopen("cc.pdb", "a");
	fprintf(fp1, "REMARK%20s\n",Iname[a]);
	for (int i = 0; i < full_len[a]; i++)
	{
		
		if (atoi(num2[a][i]) != tmp1)
		{
			tmp1 = atoi(num2[a][i]);
			count++;
		}
		if (in_or_not(count, CC_each[a]) == true)
		{
			fprintf(fp1, "%6s%5s %-4s %3s %1c%5s   %8.3f%8.3f%8.3f\n", fla[a][i], num1[a][i], ato[a][i], res[a][i],z, num2[a][i], coord[i][0], coord[i][1], coord[i][2]);
			//cout<<fla[a][i]<<" "<<num1[a][i]<<" "<<ato[a][i]<<" "<<res[a][i]<<" "<<z<<" "<<num2[a][i]<<endl;
		}
	}

	fprintf(fp1, "TER\n");
	//fprintf(fp, "%s\n","ENDMDL");
	fprintf(fp1, "\n");
	fclose(fp1);

	delete [] m1;
	delete [] m2;
	DeleteArray(&coord, full_len[a]);
	DeleteArray(&xtm, k1);
	DeleteArray(&ytm, k1);
}

void 
TMM::output_y(int loc, char z, char *out)
{
	FILE *fp = fopen(out, "a");
	int i,j;
	//fprintf(fp, "%s      %d\n","MODEL",loc);
	fprintf(fp, "REMARK%20s\n",Iname[loc]);
	for (i = 0; i < full_len[loc]; i++)
		fprintf(fp, "%6s%5s %-4s %3s %1c%5s   %8.3f%8.3f%8.3f\n", fla[loc][i] ,num1[loc][i], ato[loc][i], res[loc][i], z, num2[loc][i], full_co[loc][i][0], full_co[loc][i][1], full_co[loc][i][2]);
	for(int i=0; i < conect_len[loc]; i++)
	{
		fprintf(fp, "%s\n", conect[loc][i].c_str());
	}
	fprintf(fp, "TER\n");
	fprintf(fp, "\n");
	//fprintf(fp, "%s\n","ENDMDL");
	fclose(fp);

	vector<int> CC(get_CC_all(1));
	vector<int> CC_loc(get_CC(4, CC));

	vector< vector<int> > CC_each(tot_num, vector<int>(CC_loc.size()));
	for(int j=0; j<tot_num; j++)
	{
		for(int i=0; i<CC_loc.size(); i++)
		{
			CC_each[j][i] = Eno[j][CC_loc[i]];
		}
	}
	int tmp = atoi(num2[loc][0]);
	int count = 0;
	FILE *fp1 = fopen("cc.pdb", "a");
	fprintf(fp1, "REMARK%20s\n",Iname[loc]);
	for (int i = 0; i < full_len[loc]; i++)
	{
		if (atoi(num2[loc][i]) != tmp)
		{
			tmp = atoi(num2[loc][i]);
			count++;
		}
		if (in_or_not(count, CC_each[loc]) == true)
		{
			fprintf(fp1, "%6s%5s %-4s %3s %1c%5s   %8.3f%8.3f%8.3f\n", fla[loc][i] ,num1[loc][i], ato[loc][i], res[loc][i],z, num2[loc][i], full_co[loc][i][0], full_co[loc][i][1], full_co[loc][i][2]);
			//cout<<fla[loc][i]<<" "<<num1[loc][i]<<" "<<ato[loc][i]<<" "<<res[loc][i]<<" "<<z<<" "<<num2[loc][i]<<endl;
		}
	}
	fprintf(fp1, "TER\n");
	//fprintf(fp, "%s\n","ENDMDL");
	fprintf(fp1, "\n");
	fclose(fp1);
}

void 
TMM::output_file(int cor, char *out)
{
	char d = get_char(cor);
	
	FILE *fp = fopen(out, "w");
	FILE *fp1 = fopen("cc.pdb", "w");
	fclose(fp1);
	for(int i=0;i<cor;i++)
	{
    	char d=get_char(i);
		output_x(i,cor,max_len,d, out);
	}
	output_y(cor,d, out);
	for(int i=cor+1;i<tot_num;i++)
	{
    	char d=get_char(i);
		output_x(i,cor,max_len,d, out);
	}
	fclose(fp);
}

void 
TMM::output_x_mod(int a, int b, int kk, char *out)
{
	double t[3];
	double u[3][3];
	int max = 2000;
	int *m1,*m2;
	double **xtm, **ytm;
    double rmsd;

    double **coord;
    NewArray(&coord, full_len[a], 3);
    m1 = new int[Ilen[a]]; //alignd index in x
    m2 = new int[Ilen[b]]; //alignd index in y
    int k1=0;
    	
    vector<int> CC(get_CC_all(1));

    for(int i=0;i<CC.size();i++)
    {
        m1[k1]=Eno[a][CC[i]];
        m2[k1]=Eno[b][CC[i]];
        k1++;
    }

    NewArray(&xtm, k1, 3);
    NewArray(&ytm, k1, 3);
    for (int j = 0; j<k1; j++)
	{
		xtm[j][0] = Ico[a][m1[j]][0];
		xtm[j][1] = Ico[a][m1[j]][1];
		xtm[j][2] = Ico[a][m1[j]][2];

		ytm[j][0] = Ico[b][m2[j]][0];
		ytm[j][1] = Ico[b][m2[j]][1];
		ytm[j][2] = Ico[b][m2[j]][2];
	}

	Kabsch(xtm, ytm, k1, 1, &rmsd, t, u);
	do_rotation(full_co[a], coord, full_len[a], t, u);

	FILE *fp = fopen(out, "a");
	
	//superposed structure B
	fprintf(fp, "%s   %20s   %d\n","MODEL",Iname[a],a);
	//fprintf(fp, "REMARK%20s\n",Iname[a]);
	for (int i = 0; i < full_len[a]; i++)
	{
		fprintf(fp, "%6s%5s %-4s %3s %1c%5s   %8.3f%8.3f%8.3f\n", fla[a][i] ,num1[a][i], ato[a][i], res[a][i],'A', num2[a][i], coord[i][0], coord[i][1], coord[i][2]);
		//cout<<num1[a][i]<<" "<<ato[a][i]<<" "<<res[a][i]<<" "<<z<<" "<<num2[a][i]<<endl;
	}
	for(int i=0; i < conect_len[a]; i++)
	{
		fprintf(fp, "%s\n", conect[a][i].c_str());
	}
	//fprintf(fp, "TER\n");
	fprintf(fp, "%s\n","ENDMDL");
	fprintf(fp, "\n");
	fclose(fp);

	vector<int> CC_loc(get_CC(4, CC));
	vector< vector<int> > CC_each(tot_num, vector<int>(CC_loc.size()));
	for(int i=0; i<CC_loc.size(); i++)
	{
		for(int j=0; j<tot_num; j++)
		{
			CC_each[j][i] = Eno[j][CC_loc[i]];
		}
	}


	int tmp1 = atoi(num2[a][0]);
	int count = 0;
	FILE *fp1 = fopen("cc.pdb", "a");
	fprintf(fp, "%s   %20s   %d\n","MODEL",Iname[a],a);
	for (int i = 0; i < full_len[a]; i++)
	{
		
		if (atoi(num2[a][i]) != tmp1)
		{
			tmp1 = atoi(num2[a][i]);
			count++;
		}
		if (in_or_not(count, CC_each[a]) == true)
		{
			fprintf(fp1, "%6s%5s %-4s %3s %1c%5s   %8.3f%8.3f%8.3f\n", fla[a][i] ,num1[a][i], ato[a][i], res[a][i], 'A', num2[a][i], coord[i][0], coord[i][1], coord[i][2]);
			//cout<<num1[a][i]<<" "<<ato[a][i]<<" "<<res[a][i]<<" "<<z<<" "<<num2[a][i]<<endl;
		}
	}
	fprintf(fp1, "TER\n");
	//fprintf(fp, "%s\n","ENDMDL");
	fprintf(fp1, "\n");
	fclose(fp1);

	delete [] m1;
	delete [] m2;
	DeleteArray(&coord, full_len[a]);
	DeleteArray(&xtm, k1);
	DeleteArray(&ytm, k1);
}

void 
TMM::output_y_mod(int loc, char *out)
{
	FILE *fp = fopen(out, "a");
	int i,j;
	fprintf(fp, "%s   %20s   %d\n","MODEL",Iname[loc],loc);
	//fprintf(fp, "REMARK%20s\n",Iname[loc]);
	for (i = 0; i < full_len[loc]; i++)
		fprintf(fp, "%6s%5s %-4s %3s %1c%5s   %8.3f%8.3f%8.3f\n", fla[loc][i] ,num1[loc][i], ato[loc][i], res[loc][i], 'A', num2[loc][i], full_co[loc][i][0], full_co[loc][i][1], full_co[loc][i][2]);
	//fprintf(fp, "TER\n");
	for(int i=0; i < conect_len[loc]; i++)
	{
		fprintf(fp, "%s\n", conect[loc][i].c_str());
	}
	fprintf(fp, "%s\n","ENDMDL");
	fprintf(fp, "\n");
	fclose(fp);

	vector<int> CC(get_CC_all(1));
	vector<int> CC_loc(get_CC(4, CC));

	vector< vector<int> > CC_each(tot_num, vector<int>(CC_loc.size()));
	for(int j=0; j<tot_num; j++)
	{
		for(int i=0; i<CC_loc.size(); i++)
		{
			CC_each[j][i] = Eno[j][CC_loc[i]];
		}
	}
	int tmp = atoi(num2[loc][0]);
	int count = 0;
	FILE *fp1 = fopen("cc.pdb", "a");
	fprintf(fp, "%s   %20s   %d\n","MODEL",Iname[loc],loc);
	for (int i = 0; i < full_len[loc]; i++)
	{
		if (atoi(num2[loc][i]) != tmp)
		{
			tmp = atoi(num2[loc][i]);
			count++;
		}
		if (in_or_not(count, CC_each[loc]) == true)
		{
			fprintf(fp1, "%6s%5s %-4s %3s %1c%5s   %8.3f%8.3f%8.3f\n", fla[loc][i] ,num1[loc][i], ato[loc][i], res[loc][i], 'A', num2[loc][i], full_co[loc][i][0], full_co[loc][i][1], full_co[loc][i][2]);
			//cout<<num1[a][i]<<" "<<ato[a][i]<<" "<<res[a][i]<<" "<<z<<" "<<num2[a][i]<<endl;
		}
	}
	fprintf(fp1, "TER\n");
	//fprintf(fp, "%s\n","ENDMDL");
	fprintf(fp1, "\n");
	fclose(fp1);
}

void 
TMM::output_file_mod(int cor, char *out)
{
	FILE *fp = fopen(out, "w");
	for(int i=0;i<cor;i++)
	{
		output_x_mod(i,cor,max_len, out);
	}
	output_y_mod(cor,out);
	for(int i=cor+1;i<tot_num;i++)
	{
		output_x_mod(i,cor,max_len, out);
	}
	fclose(fp);
}

int 
TMM::first_insert(int l, int m)
{
	int ans = l;
	for(int i=0; i< max_len; i++)
	{
		if(Eno[l][i] != -1 && Eno[m][i] == -1)
		{
			ans = l;
			break;
		}
		if(Eno[l][i] == -1 && Eno[m][i] != -1)
		{
			ans = m;
			break;
		}

	}
	return ans;
}

int 	//return maximum value in vector
TMM::max_ind(vector<int> vec)	//input vector
{
	if (! vec.empty())
	{
		size_t vec_siz = vec.size();	//vector size
		int max = vec[0];	//min element
		int ind = 0;
		for (size_t i=1;i<vec_siz; i++)
		{
			if (vec[i] > max)
			{
				max = vec[i];
				ind = i;
			}
			if (vec[i] == max)
			{
				if(Ilen[i] > Ilen[ind])
				{
					ind = i;
				}
				if(Ilen[i] == Ilen[ind])
				{
					ind = first_insert(i, ind);
				}
			}
		}
		return ind;
	}
	else
	{
		return -1;
	}
}

char 
TMM::get_char(int i)
{
	if (i==0) return 'A'; 		else if (i==1) return 'B';  else if (i==2) return 'C';   else if (i==3) return 'D';  else if (i==4) return 'E';
	else if (i==5) return 'F';  else if (i==6) return 'G';  else if (i==7) return 'H';   else if (i==8) return 'I';  else if (i==9) return 'J';
	else if (i==10) return 'K'; else if (i==11) return 'L'; else if (i==12) return 'M';  else if (i==13) return 'N'; else if (i==14) return 'O';
	else if (i==15) return 'P'; else if (i==16) return 'Q'; else if (i==17) return 'R';  else if (i==18) return 'S'; else if (i==19) return 'T';
	else if (i==20) return 'U'; else if (i==21) return 'V'; else if (i==22) return 'W';  else if (i==23) return 'X'; else if (i==24) return 'Y';
	else if (i==25) return 'Z'; else if (i==26) return '['; else if (i==27) return '\\'; else if (i==28) return ']'; else if (i==29) return '^';
	else if (i==30) return '-'; else if (i==31) return '`'; else if (i==32) return 'a';  else if (i==33) return 'b'; else if (i==34) return 'c';
	else if (i==35) return 'd'; else if (i==36) return 'e'; else if (i==37) return 'f';  else if (i==38) return 'g'; else if (i==39) return 'h';
	else if (i==40) return 'i'; else if (i==41) return 'j'; else if (i==42) return 'k';  else if (i==43) return 'l'; else if (i==44) return 'm';
	else if (i==45) return 'n'; else if (i==46) return 'o'; else if (i==47) return 'p';  else if (i==48) return 'q'; else if (i==49) return 'r';
	else if (i==50) return 's'; else if (i==51) return 't'; else if (i==52) return 'u';  else if (i==53) return 'v'; else if (i==54) return 'w';
	else if (i==55) return 'x'; else if (i==56) return 'y'; else if (i==57) return 'z';	 else if (i==58) return '0'; else if (i==59) return '1';	
	else if (i==60) return '2';
}	


vector<vector<double> > 
TMM::rot_x(int loc_a, int loc_b)
{
	double t[3];
	double u[3][3];
	int *m1,*m2;
    double rmsd;
    
    double **coord;
    double **Icoord;
    double **xtm;
    double **ytm;
    
    NewArray(&Icoord, Ilen[loc_a], 3);
    for(int i=0; i< Ilen[loc_a]; i++)
    {
    	Icoord[i][0] = Ico[loc_a][i][0];
    	Icoord[i][1] = Ico[loc_a][i][1];
    	Icoord[i][2] = Ico[loc_a][i][2];
    }
    NewArray(&coord, Ilen[loc_a], 3);
    
    vector<vector<double> > tmp(Ilen[loc_a], vector<double> (3));
    m1 = new int[max_len]; //alignd index in x
    m2 = new int[max_len]; //alignd index in y
    int k1=0;
    	
    vector<int> CC(get_CC_all(1));

    for(int i=0;i<CC.size();i++)
    {
        m1[k1]=Eno[loc_a][CC[i]];
        m2[k1]=Eno[loc_b][CC[i]];
        k1++;
    }

	NewArray(&xtm, k1, 3);
    NewArray(&ytm, k1, 3);	
    for (int j = 0; j<k1; j++)
	{
		xtm[j][0] = Ico[loc_a][m1[j]][0];
		xtm[j][1] = Ico[loc_a][m1[j]][1];
		xtm[j][2] = Ico[loc_a][m1[j]][2];

		ytm[j][0] = Ico[loc_b][m2[j]][0];
		ytm[j][1] = Ico[loc_b][m2[j]][1];
		ytm[j][2] = Ico[loc_b][m2[j]][2];
	}
	
	Kabsch(xtm, ytm, k1, 1, &rmsd, t, u);
	do_rotation(Icoord, coord, Ilen[loc_a], t, u);

	for(int i=0; i< Ilen[loc_a]; i++)
	{
		tmp[i][0] = coord[i][0];
		tmp[i][1] = coord[i][1];
		tmp[i][2] = coord[i][2];
	}
	
	DeleteArray(&xtm, k1);
	DeleteArray(&ytm, k1);
	DeleteArray(&coord, Ilen[loc_a]);
	DeleteArray(&Icoord, Ilen[loc_a]);
	delete [] m1;
	delete [] m2;
	return tmp;
}


int 
TMM::select_core()
{
	vector<int> vec_len_ali(tot_num);
	for(int i=0; i< tot_num; i++)
	{
		int count = 0;
		for(int j=0; j< max_len; j++)
		{
			if(Eno[i][j] != -1)
			{
				bool flag = false;
				for(int k=0; k< tot_num; k++)
				{
					if(Eno[k][j] != -1 && k != i)
					{
						flag = true;
						continue;
					}
				}
				if (flag == true)
					count ++;
			}
		}
		vec_len_ali[i] = count;
	}
	return max_ind(vec_len_ali);
}

vector<int> 
TMM::get_CC_all(double percent)
{
	double cut_num = tot_num * percent;
	vector<int> ans;
	for(int i=0; i< max_len; i++)
	{
		int count = 0;
		for(int j=0; j< tot_num; j++)
		{
			if(Eno[j][i] != -1)
			{
				count++;
			}
		}
		if(count >= cut_num)
		{
			ans.push_back(i);
		}
	}
	return ans;
}

void 
TMM::show_inf()
{
	string version = "20180725";
	cout <<endl;

	cout << " ***************************************************************" << endl;
	cout << " *                 mTM-align (Version "<< version <<")                *" << endl;
	cout << " * An algorithm for multiple protein structure alignment (MSTA)*"<<endl;
	cout << " * Reference: Dong, et al, Bioinformatics, 34: 1719-1725 (2018)*"<<endl;
	cout << " * Please email your comments to: yangjy@nankai.edu.cn         *"<<endl;
	cout << " ***************************************************************" << endl;

	cout << endl;
	cout<<endl;
	cout<<"------------- Summary of your input -------------"<<endl;
	//cout<<endl;
	cout<<"Input "<<Iname.size()<<" structures: ";
	for (unsigned int i = 0; i < tot_num-1; i++)
	{
		string name1=Iname[i];
		cout << name1 << ", ";	//structure name
	}
	cout << Iname[tot_num-1] << endl;  	
	double ave_len = 0;
	for (unsigned int i = 0; i < tot_num; i++)
	{
		ave_len += Ilen[i];
	}
	ave_len = ave_len / tot_num;
	cout<<"Average length: "<<ave_len<<endl;
	cout<<endl;
}

template<class T> T 	//return maximum value in vector
TMM::max_of_vector(vector<T> vec)	//input vector
{
	if (! vec.empty())
	{
		size_t vec_siz = vec.size();	//vector size
		T max = vec[0];	//min element
		for (size_t i=0;i<vec_siz; i++)
		{
			if (vec[i] > max)
			{
				max = vec[i];
			}
		}
		return max;
	}
	else
	{
		return 0;
	}
}

void	//save result to seq.txt
TMM::show_seq()
{

	cout << endl;
	cout<<"------------- MSTA  -------------"<<endl;
	//cout<<endl;
	vector<int> vec_name_len(tot_num);
	for (int i=0; i< tot_num; i++)
	{
		string name1 = Iname[i];
		vec_name_len[i] = name1.length();
	}
	int max_name_len = max_of_vector(vec_name_len);

	for (unsigned int i = 0; i < tot_num; i++)
	{
		string name1=Iname[i];
		cout << name1;
		for (int k=0; k< max_name_len-vec_name_len[i]; k++)
		{
			cout<<" ";
		}
		cout<< '\t';	//structure name
		for (int j = 0; j < Elen[0]; j++)
		{
			
			if (Eno[i][j] != -1)
			{
				cout << Iseq[i][Eno[i][j]];
			}
			else
			{
				if (Eno[i][j] == -1)
				{
					cout << '-';
				}
			}
		}
		cout << endl;
	}
	cout<<endl;
}

void 
TMM::get_new_co()
{
	cor = select_core();
	for(int i=0;i<tot_num;i++)
	{
		if (i != cor)
		{
			vector<vector<double> > tmp(rot_x(i, cor));
			new_co.push_back(tmp);
		}
		if (i == cor)
		{
			new_co.push_back(Ico[cor]);
		}
	}
}

double 
TMM::get_dis(vector<double> x, vector<double> y)
{
	double dis = sqrt((x[0]-y[0])*(x[0]-y[0]) + (x[1]-y[1])*(x[1]-y[1]) +(x[2]-y[2])*(x[2]-y[2]));
	return dis;
}

void 
TMM::get_dis_vec()
{
	
	for (int i=0; i< tot_num; i++)
	{
		for(int j=i+1; j< tot_num; j++)
		{
			vector<double> tmp(max_len);
			for(int l=0; l< max_len; l++)
			{
				if(Eno[i][l] != -1 && Eno[j][l] != -1) 
				{
					tmp[l] = get_dis(new_co[i][Eno[i][l]], new_co[j][Eno[j][l]]);
				}
				else
				{
					tmp[l] = -1;
				}
			}
			vec_dis.push_back(tmp);
		}
	}
}

vector<int> 
TMM::get_CC(double cut_off, vector<int> loc)
{
	int tot_num_pair = tot_num * (tot_num - 1) / 2;
	int count = 0;
	vector<int> CC_loc;
	for(int i=0; i< loc.size(); i++)
	{
		bool flag = false;
		for(int j=0; j< tot_num_pair; j++)
		{
			if(vec_dis[j][loc[i]] > cut_off)
			{
				flag = true;
				break;
			}
		}
		if(flag == false)
		{
			CC_loc.push_back(loc[i]);
		}
	}
	return CC_loc;
}

double 
TMM::single_RMSD(int a,int b, vector<int> CC_loc)
{	
	double **r1, **r2;
	double rmsd;
	double t[3];
	double u[3][3];
	int kk = 0;
	NewArray(&r1, CC_loc.size(), 3);
	NewArray(&r2, CC_loc.size(), 3);
	
	for (int i = 0; i < CC_loc.size();i++)
	{
		if(Eno[a][CC_loc[i]] != -1 && Eno[b][CC_loc[i]] != -1)
		{
			r1[kk][0] = Ico[a][Eno[a][CC_loc[i]]][0];
			r1[kk][1] = Ico[a][Eno[a][CC_loc[i]]][1];
			r1[kk][2] = Ico[a][Eno[a][CC_loc[i]]][2];

			r2[kk][0] = Ico[b][Eno[b][CC_loc[i]]][0];
			r2[kk][1] = Ico[b][Eno[b][CC_loc[i]]][1];
			r2[kk][2] = Ico[b][Eno[b][CC_loc[i]]][2];
			kk++;
		}
	}
	
	Kabsch(r1, r2, kk, 1, &rmsd, t, u);
	rmsd = sqrt(rmsd/kk);
	DeleteArray(&r1, CC_loc.size());
	DeleteArray(&r2, CC_loc.size());
	return rmsd;
}

double 
TMM::multi_RMSD(vector<int> CC_loc)
{
	double sum_RMSD = 0;
	double a;
	for (int i = 0; i < tot_num;i++)
	{
		for (int j = i + 1; j < tot_num;j++)
		{
			a = single_RMSD(i, j, CC_loc);
			if(a >= 0)
			{
				sum_RMSD += a;
			}
		}
	}
	sum_RMSD = sum_RMSD / (tot_num*(tot_num-1)/2);
	return sum_RMSD;
}

vector<vector<int> >
TMM::get_n_CC_seq(vector<int> CC_loc)
{
	vector<vector<int> > n_CC_seq(tot_num, vector<int> (CC_loc.size()));
	for(int i=0; i< tot_num; i++)
	{
		for(int j=0; j< CC_loc.size(); j++)
		{
			n_CC_seq[i][j] = Eno[i][CC_loc[j]];
		}
	}
	return n_CC_seq;
}

double 
TMM::get_average_cc_TM_score(vector<int> CC_loc)
{
	double TM=0;
	int tot_num_pair=tot_num*(tot_num-1)/2;
	vector<vector<int> > n_CC_seq(get_n_CC_seq(CC_loc));
	for(int i=0;i<tot_num;i++)
	{
		for(int j=0;j<i;j++)
		{
			double temp=get_TMscore(Iname[i],Iname[j],n_CC_seq[i],n_CC_seq[j],CC_loc.size());
			TM+=temp;
		}
	}
	TM=TM/tot_num_pair;
	return TM;
}

double 
TMM::get_TMscore(char *a,char *b,vector<int> noxA,vector<int> noyA, int kk)
{  
    int *m1,*m2;
    
    load_PDB_allocate_memory(a,b);
    m1 = new int[xlen]; //alignd index in x
    m2 = new int[ylen]; //alignd index in y
    int k1=0;
    	
    for(int i=0;i<kk;i++)
    {
        if(noxA[i]!=-1 && noyA[i]!=-1)
        {
            m1[k1]=noxA[i];
            m2[k1]=noyA[i];
            k1++;
        }
    }
   
    parameter_set4search(xlen,ylen);
    for (int j = 0; j<k1; j++)
	{
		xtm[j][0] = xa[m1[j]][0];
		xtm[j][1] = xa[m1[j]][1];
		xtm[j][2] = xa[m1[j]][2];

		ytm[j][0] = ya[m2[j]][0];
		ytm[j][1] = ya[m2[j]][1];
		ytm[j][2] = ya[m2[j]][2];
	}
    double rmsd, TM1, TM2;
    int simplify_step = 1;
    int score_sum_method = 0;
	
    double t0[3], u0[3][3];
    double d0_0, TM_0;
    double Lnorm_0;

    if (xlen >= ylen)
    {
        Lnorm_0=ylen;
        parameter_set4final(Lnorm_0);
        d0_0 = d0;
        TM_0 = TMscore8_search(xtm, ytm, k1, t0, u0, simplify_step, score_sum_method, &rmsd);
	
    }
	
    if (xlen  < ylen)
    {
        Lnorm_0=xlen;
        parameter_set4final(Lnorm_0);
        d0_0 = d0;
        TM_0 = TMscore8_search(xtm, ytm, k1, t0, u0, simplify_step, score_sum_method, &rmsd);
    }

	
	delete [] m1;
	delete [] m2;
    return TM_0;
	
}

int 
TMM::common_length(int a,int b)
{
	int count = 0;
	for (int i = 0; i < max_len; i++)
	{
		if (Eno[a][i] != -1 && Eno[b][i] != -1)
		{
			count++;
		}
	}
	return count;
}

vector<int> 
TMM::caculate_pc(int a, int b, double cut_off)
{
	int c = common_length(a, b);
	double **r1, **r2;
	double rmsd;
	double t[3];
	double u[3][3];
	int kk = 0;
	NewArray(&r1, c, 3);
	NewArray(&r2, c, 3);
	double **coord;
	double **coord1;
	NewArray(&coord, Ilen[a], 3);
	NewArray(&coord1, Ilen[a], 3);
	
	for (int i = 0; i < max_len;i++)
	{
		if (Eno[a][i]!=-1 && Eno[b][i]!=-1)
		{
			r1[kk][0] = Ico[a][Eno[a][i]][0];
			r1[kk][1] = Ico[a][Eno[a][i]][1];
			r1[kk][2] = Ico[a][Eno[a][i]][2];

			r2[kk][0] = Ico[b][Eno[b][i]][0];
			r2[kk][1] = Ico[b][Eno[b][i]][1];
			r2[kk][2] = Ico[b][Eno[b][i]][2];
			kk++;
		}
	}
	
	for(int i=0; i< Ilen[a];i++)
	{
		coord1[i][0] = Ico[a][i][0];
		coord1[i][1] = Ico[a][i][1];
		coord1[i][2] = Ico[a][i][2];
	}
	Kabsch(r1, r2, c, 1, &rmsd, t, u);
	do_rotation(coord1, coord, Ilen[a], t, u);
	int count = 0;

	vector<int> ans;
	for (int i = 0; i < max_len;i++)
	{
		if (Eno[a][i]!=-1 && Eno[b][i]!=-1)
		{
			double *tmp;
			tmp = new double[3];
			for (int j=0; j< 3; j++)
			{
				tmp[j] = Ico[b][Eno[b][i]][j];
			}
			double dis = sqrt(dist(tmp, coord[Eno[a][i]]));
			if(dis < cut_off)
			{
				ans.push_back(i);
			}
		}
	}
	DeleteArray(&r1, c);
	DeleteArray(&r2, c);
	DeleteArray(&coord, Ilen[a]);
	DeleteArray(&coord1, Ilen[a]);
	return ans;
}

double 
TMM::average_pc(double cut_off)
{
	double count = 0;
	int tot_num_pair = tot_num*(tot_num-1)/2;
	for(int i=0; i< tot_num; i++)
	{
		for(int j=i+1; j< tot_num; j++)
		{
			if (Ilen[i] >= Ilen[j])
			{
				vector<int> tmp(caculate_pc(i, j, cut_off));
				count += tmp.size();
			}
			if (Ilen[i] < Ilen[j])
			{
				vector<int> tmp(caculate_pc(j, i, cut_off));
				count += tmp.size();
			}
		}
	}
	count = count/tot_num_pair;
	return count;
}
	
double 
TMM::single_pc_RMSD(int a,int b, double cut_off)
{
	vector<int> pc(caculate_pc(a, b, cut_off));
	int c = pc.size();
	double **r1, **r2;
	double rmsd;
	double t[3];
	double u[3][3];
	int kk = 0;
	NewArray(&r1, c, 3);
	NewArray(&r2, c, 3);

	for (int i = 0; i < c;i++)
	{
		r1[kk][0] = Ico[a][Eno[a][pc[i]]][0];
		r1[kk][1] = Ico[a][Eno[a][pc[i]]][1];
		r1[kk][2] = Ico[a][Eno[a][pc[i]]][2];

		r2[kk][0] = Ico[b][Eno[b][pc[i]]][0];
		r2[kk][1] = Ico[b][Eno[b][pc[i]]][1];
		r2[kk][2] = Ico[b][Eno[b][pc[i]]][2];
		kk++;
	}
	Kabsch(r1, r2, c, 0, &rmsd, t, u);
	rmsd = sqrt(rmsd/c);
	DeleteArray(&r1, c);
	DeleteArray(&r2, c);
	return rmsd;
}
double 
TMM::multi_pc_RMSD()
{
	double sum_RMSD = 0;
	double a;
	for (int i = 0; i < tot_num;i++)
	{
		for (int j = i + 1; j < tot_num;j++)
		{
			a = single_pc_RMSD(i, j, 4);
			if(a >= 0)
			{
				sum_RMSD += a;
			}
		}
	}
	sum_RMSD = sum_RMSD / (tot_num*(tot_num-1)/2);
	return sum_RMSD;
}

double 
TMM::get_average_TM_score(double cut_off)
{
	double TM=0;
	int number=tot_num*(tot_num-1)/2;

	for(int i=0;i<tot_num;i++)
	{
		for(int j=0;j<i;j++)
		{
			double temp=get_TMscore_from_seqxA(i,j,Iname[i],Iname[j],Eno[i],Eno[j],cut_off);
			TM=TM+=temp;
		}
	}

	TM=TM/number;
	return TM;
}

void 
TMM::print_help(char *arg)
{
	string version = "20180725";
	cout <<endl;
	cout << " ***************************************************************" << endl;
    cout << " *                 mTM-align (Version "<< version <<")                *" << endl;
    cout << " * An algorithm for multiple protein structure alignment (MSTA)*"<<endl;
    cout << " * Reference: Dong, et al, Bioinformatics, 34: 1719-1725 (2018)*"<<endl;
    cout << " * Please email your comments to: yangjy@nankai.edu.cn         *"<<endl;
    cout << " ***************************************************************"<< endl;
	cout <<endl;


	cout << " Usage: " << arg << " -i <input_list> [Options]" << endl;
	cout << " Options:" << endl;
	cout << "   -i input_list   The input_list is an input file, listing the file names of the structures to be aligned."<< endl;
	cout << "                   Each line represents the file name for one structure." <<endl;
	cout << "                   Please note that each input structure should be a single-chain structure."<<endl<<endl;
	cout << "   -o filename     The name of the file to save the superimposed structures. The default is 'result.pdb'" <<endl;
	cout << "                   When the number of input structures is >61, the superimposed structures will be separated by 'MODEL'" << endl;
	cout << "                   Otherwise, the structures are speparated using the chain IDs: A,B,C,...\n" << endl<<endl;
	cout << "  -v               Print the version of mTM-align" << endl;
	cout << "  -h               Print this help information" << endl;
	cout << " Example usage:" << endl;
	cout << " "<< arg <<" -i input_list" << endl;
	cout << " "<< arg <<" -i input_list -o result.pdb" << endl;
	   
	exit(EXIT_SUCCESS);
}

double 
TMM::get_TMscore_from_seqxA(int aa, int bb, char *a,char *b,vector<int> noxA,vector<int> noyA, double cut_off)
{  
    int *m1,*m2;
    vector<int> pc(caculate_pc(aa, bb, cut_off));
	int c = pc.size();

    load_PDB_allocate_memory(a,b);
    m1 = new int[c]; //alignd index in x
    m2 = new int[c]; //alignd index in y
    int k1=0;
    	

    for(int i=0;i<c;i++)
    {
        m1[k1]=noxA[pc[i]];
        m2[k1]=noyA[pc[i]];
        k1++;
    }
   
    parameter_set4search(xlen,ylen);
    for (int j = 0; j<c; j++)
	{
		xtm[j][0] = xa[m1[j]][0];
		xtm[j][1] = xa[m1[j]][1];
		xtm[j][2] = xa[m1[j]][2];

		ytm[j][0] = ya[m2[j]][0];
		ytm[j][1] = ya[m2[j]][1];
		ytm[j][2] = ya[m2[j]][2];
	}
    double rmsd, TM1, TM2;
    int simplify_step = 1;
    int score_sum_method = 0;
	
    double t0[3], u0[3][3];
    double d0_0, TM_0;
    double Lnorm_0;

    if (xlen >= ylen)
    {
        Lnorm_0=ylen;
        parameter_set4final(Lnorm_0);
        d0_0 = d0;
        TM_0 = TMscore8_search(xtm, ytm, c, t0, u0, simplify_step, score_sum_method, &rmsd);
	
    }
	
    if (xlen  < ylen)
    {
        Lnorm_0=xlen;
        parameter_set4final(Lnorm_0);
        d0_0 = d0;
        TM_0 = TMscore8_search(xtm, ytm, k1, t0, u0, simplify_step, score_sum_method, &rmsd);
    }

	
	delete [] m1;
	delete [] m2;
    return TM_0;
	
}
