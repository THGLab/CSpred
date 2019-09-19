
#include <string>
#include <vector>

using namespace std;

class UPGMA_node
{
public:
	UPGMA_node create_node(string a, int b);
	string get_node(UPGMA_node a);
	int get_node_level(UPGMA_node a);
private:
	string node_name;
	int node_level;
};

class UPGMA_edge: public UPGMA_node
{
public:
	UPGMA_edge create_edge(UPGMA_node A, UPGMA_node B, double c);
	pair<string,string> get_two_node(UPGMA_edge a);
	pair<int,int> get_two_level(UPGMA_edge a);
	double get_distance(UPGMA_edge a);
private:
	UPGMA_node node1;
	UPGMA_node node2;
	double distance_edge;
};

class UPGMA : public UPGMA_edge
{
public:
	template <class A> void Newarray(A *** array, int Narray1, int Narray2)
	{
		*array=new A* [Narray1];
		for(int i=0; i<Narray1; i++) *(*array+i)=new A [Narray2];
	};

	template <class A> void Deletearray(A *** array, int Narray)
	{
		for(int i=0; i<Narray; i++)
			if(*(*array+i)) delete [] *(*array+i);
				if(Narray) delete [] (*array);
		(*array)=NULL;
	};
	void read_file(string a);
	int get_loc_of_name(string name);
	pair<int, int> min_distance();
	void change_level(UPGMA_node a, int b);
	double get_distance_from_matrix(string a, string b);
	bool belong_or_not(string *a, string b);
	bool belong_or_not_vector(vector<string> a, string b);
	vector<string> segmentation(string a);
	double get_distance_of_edge(UPGMA_node a, UPGMA_node b);
	bool exit_of_edge(UPGMA_node a, UPGMA_node b);
	double get_distance_of_two_node(string a, string b);
	double get_distance_to_bottom(string a);
	double get_distance_from_begin(string a, string b);
	void renewal_matrix(int column, int row);
	void construction_of_tree();
	vector<pair<string, string> > find_alignment_target();
	
private:
	vector<UPGMA_node> node_of_tree;
	vector<UPGMA_edge> edge_of_tree;
	double **distance_matrix;
	double **distance_matrix_begin;
	string *structure_name;
	string *structure_name_begin;
	int matrix_len;
	int max_len;
};
