#include "UPGMA.h"
#include <ctime>

class TMM
{
public:
	void matrix_output(vector<char*> vec_input);
	vector<string> segmentation(string astr);
	double get_score_from_matrix(int b,int c,int d);
	int get_Element_location(string nameE);
	int get_Initial_location(string nameI);
	int get_Pair_location(string name1,string name2);
	double caculate_distance_square(vector<double> ads, vector<double> bds);
	vector<vector<double> > caculate_dis_matrix(int len_x, int len_y, vector<vector<double> > u, vector<double> t, vector<vector<double> > co_x, vector<vector<double> > co_y);
	vector<vector<double> > caculate_sco_matrix(int len_x, int len_y, vector<vector<double> > matrix_dis, vector<char> seq_x, vector<char> seq_y, double d0, double TMscore);
	double max_of_two(double a,double b);
	double min_of_two(double a, double b);
	vector<vector<double> > get_score_matrix(string a,string b);	
	vector<int> NWDP_Nor(int len_1, int len_2, vector<vector<double> > score_matrix);
	pair<vector<vector<int> >, int> get_first_alignment(string afi, string bfi);
	void update_E(string str_x, string str_y, vector<vector<int> > vec_no, vector<int> vec_ind, int ali_len);
	pair<vector<vector<int> >, int> regular_number(string str_x, string str_y, vector<int> ind_x, vector<int> ind_y, vector<vector<int> > no_x, vector<vector<int> > no_y, vector<vector<double> > score_matrix);
	void save_seq();
	bool exist_in_initial(string a_base);
	vector<int> Align_ini(pair<string, string> aa);
	vector<int> Align_unini(pair<string, string> aa, vector<vector<double> > sco);
	vector<int> Align_one_step(pair<string, string> aa, vector<vector<double> > sco);
	void Align(vector<pair<string, string> > vec_ord);
	void programming(vector<char*> vec_input, char *out);
	double get_average_pair_TMscore();
	double get_Blosum62(char a, char b);
	vector<vector<double> >	caculate_sco_matrix_1(int len_x, int len_y, vector<vector<double> > u, vector<double> t, vector<vector<double> > co_x, vector<vector<double> > co_y, vector<char> seq_x, vector<char> seq_y, double d0);
	void load_all();
	void output_pair(char *xname, char *yname, int a, int b, double t0[3], double u0[3][3]);
	void output_x(int a, int b, int kk, char z, char *out);
	void output_y(int loc, char z, char *out);
	void output_file(int cor, char *out);
	void output_x_mod(int a, int b, int kk, char *out);
	void output_y_mod(int loc, char *out);
	void output_file_mod(int cor, char *out);
	int first_insert(int l, int m);
	int max_ind(vector<int> vec);
	int select_core();
	vector<int> get_CC_all(double percent);
	void show_inf();
	template<class T> T max_of_vector(vector<T> vec);
	void show_seq();
	void get_new_co();
	double get_dis(vector<double> x, vector<double> y);
	void get_dis_vec();
	vector<int> get_CC(double cut_off, vector<int> loc);
	double single_RMSD(int a,int b, vector<int> CC_loc);
	double multi_RMSD(vector<int> CC_loc);
	char get_char(int i);
	vector<vector<int> > get_n_CC_seq(vector<int> CC_loc);
	double get_average_cc_TM_score(vector<int> CC_loc);
	int common_length(int a,int b);
	vector<int> caculate_pc(int a, int b, double cut_off);
	double average_pc(double cut_off);
	double single_pc_RMSD(int a,int b, double cut_off);
	double multi_pc_RMSD();
	double get_TMscore(char *a,char *b,vector<int> noxA,vector<int> noyA, int kk);
	double get_average_TM_score(double cut_off);
	void print_help(char *arg);
	double get_TMscore_from_seqxA(int aa, int bb, char *a,char *b,vector<int> noxA,vector<int> noyA, double cut_off);
vector<vector<double> > rot_x(int loc_a, int loc_b);

private:
	int max_len;	//aligned len
	int tot_num;	//total number of structures
	UPGMA Tree;	//phylogenetic tree
};


