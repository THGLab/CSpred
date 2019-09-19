//construction of phylogenetic tree by UPGMA
#include "UPGMA.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdlib.h>
#include <string.h> 
#include <stdio.h> 
#include <iomanip> 
UPGMA_node  //create a node with name and level 
UPGMA_node::create_node(string a,	//node name 
						int b)	//node level
{
	UPGMA_node A;
	A.node_name = a;
	A.node_level = b;
	return A;
}

string  //get name from the node
UPGMA_node::get_node(UPGMA_node A)	//the node
{
	return A.node_name;
}

int  //get level from the node
UPGMA_node::get_node_level(UPGMA_node A)	//the node
{
	return A.node_level;
}

UPGMA_edge   //create a edge with node A ,node B and distance c
UPGMA_edge::create_edge(UPGMA_node A, //node1
						UPGMA_node B, //node2
						double c)	//the distance between node1 and node2
{
	UPGMA_edge a;
	if ((get_node(A)).length() <= (get_node(B)).length())
	{
		a.node1 = A;
		a.node2 = B;
		a.distance_edge = c;
	}
	else if ((get_node(A)).length() > (get_node(B)).length())
	{
		a.node1 = B;
		a.node2 = A;
		a.distance_edge = c;
	}
	return a;
}

pair<string,string> //get two nodes in egde a, sotre them in pair
UPGMA_edge::get_two_node(UPGMA_edge a)	//the edge
{
	string node_1=get_node(a.node1);
	string node_2=get_node(a.node2);
	pair<string,string> ans(node_1,node_2);
	return ans;
}

pair<int,int> //get level of two nodes in egde a, sotre them in pair
UPGMA_edge::get_two_level(UPGMA_edge a)	//the edge
{
	int level_1 = get_node_level(a.node1);
	int level_2 = get_node_level(a.node2);
	pair<int,int> ans(level_1,level_2);
	return ans;
}

double //get distance of egde a
UPGMA_edge::get_distance(UPGMA_edge a)	//the edge
{
	return a.distance_edge;
}

void	//read matrix file, store distance and structure name in arrays
UPGMA::read_file(string matrix_file)	//the matrix file
{
	vector<string> matrix_text; //save word in text 
	ifstream infile;
	infile.open(matrix_file.c_str());	//open matrix file
	if (!infile)
		cerr << "matrix.txt error!" << endl;
	istream_iterator<string> is(infile.seekg(0));
	istream_iterator<string> eof;
	for (; is != eof; is++)
	{
		matrix_text.push_back(*is);
	}
	int len = atof(matrix_text[0].c_str());
	matrix_len = len;
	max_len = len;
	Newarray(&distance_matrix, len, len);	//save variable distance matrix
	Newarray(&distance_matrix_begin, len, len);	//save initial distance matrix
	
	structure_name = new string[len];	//save node name
	structure_name_begin = new string[len];	//save initial structure name
	for (unsigned int i = 0; i < len; i++)
	{
		structure_name[i] = matrix_text[1 + i*(len + 1)];	
		structure_name_begin[i] = structure_name[i];	

	}
	for (unsigned int i = 0; i < len; i++)
	{
		for (unsigned int j = 0; j < len; j++)
		{
			distance_matrix[i][j] = atof(matrix_text[2+j+i*(len+1)].c_str());
			distance_matrix_begin[i][j] = distance_matrix[i][j];
		}
	}
}

int 	//input structure name, return the location of the name in structure_name array 
UPGMA::get_loc_of_name(string name)	//the structure name
{
	int loc = -1;
	for(int i=0; i< max_len; i++)
	{
		if(structure_name_begin[i] == name)
		{
			loc = i;
			break;
		}
	}
	return loc;
}

pair<int, int>	//find minimum value in distance matrix, return column and row of the value 
UPGMA::min_distance()
{
	double min_dis = 10000;	//save minimum value
	int column = -1;	//save column
	int row = -1;	//save row

	for (unsigned int i = 0; i < matrix_len; i++)
	{
		for (unsigned int j = 0; j < i; j++)
		{
			if (distance_matrix[i][j] < min_dis)
			{
				min_dis = distance_matrix[i][j];
				column = i;
				row = j;
			}
		}
	}
	pair<int, int> ans(column, row);
	return ans;
}

void	//change node level 
UPGMA::change_level(UPGMA_node A, //the node whose level needs to be changed
					int b)	//new level
{
	for (unsigned int i = 0; i < node_of_tree.size(); i++)	//change node level in node_of_tree
	{
		if (get_node(node_of_tree[i]) == get_node(A))
		{
			node_of_tree.erase(node_of_tree.begin() + i);
			node_of_tree.insert(node_of_tree.begin() + i, create_node(get_node(A), b));
		}
	}
	for (unsigned int i = 0; i < edge_of_tree.size(); i++)	//change node level in edge_of_tree
	{
		if (get_two_node(edge_of_tree[i]).first == get_node(A))
		{
			UPGMA_node C = create_node(get_two_node(edge_of_tree[i]).second, get_two_level(edge_of_tree[i]).second);
			double d = get_distance(edge_of_tree[i]);
			UPGMA_edge f = create_edge(create_node(get_node(A), b), C, d);
			edge_of_tree.erase(edge_of_tree.begin() + i);
			edge_of_tree.insert(edge_of_tree.begin() + i, f);
		}
		if (get_two_node(edge_of_tree[i]).second == get_node(A))
		{
			UPGMA_node C = create_node(get_two_node(edge_of_tree[i]).first, get_two_level(edge_of_tree[i]).first);
			double d = get_distance(edge_of_tree[i]);
			UPGMA_edge f = create_edge(C, create_node(get_node(A), b), d);
			edge_of_tree.erase(edge_of_tree.begin() + i);
			edge_of_tree.insert(edge_of_tree.begin() + i, f);
		}
	}
}

double	//input structure name a and b, return distance in distance matrix between them 
UPGMA::get_distance_from_matrix(string a,	//structure name a 
								string b)	//structure name b
{
	int max = -1;
	int min = -1;
	int c = -1;
	int d = -1;
	for (unsigned int i = 0; i < matrix_len; i++)	//find the location of a and b
	{
		if (structure_name[i] == a)
		{
			c = i;
			break;
		}
	}
	for (unsigned int i = 0; i < matrix_len; i++)
	{
		if (structure_name[i] == b)
		{
			d = i;
			break;
		}
	}
	if (c>d)
	{
		max = c;
		min = d;
	}
	if (c<d)
	{
		max = d;
		min = c;
	}
	double ans = distance_matrix[max][min];
	return ans;
}

bool //determine whether string b belongs to array a 
UPGMA::belong_or_not(string *a,	//string array a 
					string b)	//string b
{
	bool flag = false;
	for (unsigned int i = 0; i < matrix_len; i++)
	{
		if (a[i]==b)
		{
			flag = true;
			break;
		}
	}
	return flag;
}

bool	//determine whether string b belongs to vector a
UPGMA::belong_or_not_vector(vector<string> a,	//string vector a 
							string b)	//string b
{
	bool flag = false;
	for (unsigned int i = 0; i < a.size(); i++)
	{
		if (a[i] == b)
		{
			flag = true;
			break;
		}
	}
	return flag;
}

vector<string> //cut string a by spance, save them in vector
UPGMA::segmentation(string a)//string a
{
	vector<string> c;
	int j = 0;
	for (unsigned int i = 0; i < a.length(); i++)
	{
		if (a[i] == '+')
		{
			string b;
			b = a.substr(j, i - j);
			j = i + 1;
			c.push_back(b);
		}
		if (i == a.length() - 1)
		{
			string b;
			b = a.substr(j, i - j + 1);
			c.push_back(b);
		}
	}
	return c;
}

double  //input node a and b, return the distance edge consited by a and b
UPGMA::get_distance_of_edge(UPGMA_node A,	//node A 
							UPGMA_node B)	//node B
{
	double c;
	for (unsigned int i = 0; i < edge_of_tree.size(); i++)	//find edge
	{
		if (get_node(A) == get_two_node(edge_of_tree[i]).first && get_node(B)==get_two_node(edge_of_tree[i]).second)
		{
			c = get_distance(edge_of_tree[i]);
		}
		if (get_node(A) == get_two_node(edge_of_tree[i]).second && get_node(B)==get_two_node(edge_of_tree[i]).first)
		{
			c = get_distance(edge_of_tree[i]);
		}
	}
	return c;
}

bool	//determine whtether there is an edge between node a and b
UPGMA::exit_of_edge(UPGMA_node A,	//node A 
					UPGMA_node B)	//node B
{
	bool c = false;
	for (unsigned int i = 0; i < edge_of_tree.size(); i++)	//find edge
	{
		if (get_node(A) == get_two_node(edge_of_tree[i]).first && get_node(B)==get_two_node(edge_of_tree[i]).second)
		{
			c = true;
			break;
		}
		if (get_node(B) == get_two_node(edge_of_tree[i]).first && get_node(A)==get_two_node(edge_of_tree[i]).second)
		{
			c = true;
			break;
		}
	}
	return c;
}

double //input node name a and b, return the distance between them from tree
UPGMA::get_distance_of_two_node(string a, //node name a
								string b)	//node name b
{
	double dis = 0;
	string son, father;
	vector<UPGMA_node> c;	//save node on the path way
	if (a.length() <= b.length())	//judge who is father node
	{
		son = a;
		father = a;
	}
	if (a.length()>b.length())
	{
		son = b;
		father = a;
	}
	UPGMA_node son_node, father_node;
	for (unsigned int i = 0; i < node_of_tree.size();i++)	//find the location of son node and father node in node_of_tree
	{
		if (get_node(node_of_tree[i]) == son)
		{
			son_node = node_of_tree[i];
		}
	}
	for (unsigned int i = 0; i < node_of_tree.size(); i++)
	{
		if (get_node(node_of_tree[i]) == father)
		{
			father_node = node_of_tree[i];
		}
	}
	c.push_back(son_node);
	int i = 0;
	while (get_node(c.back()) != father)	//find the way from son node to father node
 	{
		if (exit_of_edge(c.back(), node_of_tree[i]) == true && (get_node(node_of_tree[i])).length() > (get_node(c.back())).length())
		{
			dis += get_distance_of_edge(c.back(), node_of_tree[i]);
			c.push_back(node_of_tree[i]);
		}
		i++;
	}
	return dis;
}

double //input node name a, return the distance from the node to bottom of the tree
UPGMA::get_distance_to_bottom(string a)	// node name a
{
	vector<string> aa = segmentation(a);
	double dis = get_distance_of_two_node(aa[0], a);
	return dis;
}

double //input node name a and b, caculate the distance between them from initial distance matrix
UPGMA::get_distance_from_begin(string a,	//node name a 
								string b)	//node name b
{
	vector<string> aa (segmentation(a));
	vector<string> bb (segmentation(b));
	double ans = 0;
	for (unsigned int i = 0; i < max_len; i++)
	{
		for (unsigned int j = 0; j < i; j++)
		{
			if (belong_or_not_vector(aa, structure_name_begin[i]) == true && belong_or_not_vector(bb, structure_name_begin[j]) == true)
			{
				ans = ans + distance_matrix_begin[i][j];
			}
			if (belong_or_not_vector(aa, structure_name_begin[j]) == true && belong_or_not_vector(bb, structure_name_begin[i]) == true)
			{
				ans = ans + distance_matrix_begin[i][j];
			}
		}
	}
	ans = ans / (aa.size()*bb.size());
	return ans;
}

void //input the column and row of minimum value in distance matrix, renew distance matrix after merge node
UPGMA::renewal_matrix(int column,	//the column of minimum value in distance matrix
						int row)	//the row of minimum value in distance matrix
{
	int max = -1;
	int min = -1;
	if (column>row)
	{
		max = column;
		min = row;
	}
	if (column < row)
	{
		max = row;
		min = column;
	}

	int matrix_len_renewal = matrix_len - 1;	//new size of new distance matrix
	
	string *standby1 = new string[matrix_len_renewal];	//new node name array
	
	unsigned int j = 0;
	for (unsigned int i = 0; i < matrix_len; i++)	//renewl node name array
	{
		if (j != matrix_len_renewal - 1)	
		{
			if (i!=min && i!=max)
			{
				standby1[j] = structure_name[i];
				j = j + 1;
			}
		}
		if (j == matrix_len_renewal - 1)
		{
			standby1[j] = structure_name[max] + "+" + structure_name[min];
		}
	}

	double **standby;	//new distance matrix 
	Newarray(&standby, matrix_len_renewal, matrix_len_renewal);

	for (unsigned int i = 0; i < matrix_len_renewal; i++)	//initialize
	{
		for (unsigned int j = 0; j < matrix_len_renewal; j++)
		{
			standby[i][j] = 0;
		}
	}

	for (unsigned int i = 0; i < matrix_len_renewal; i++)	//comstruction of new distance matrix
	{
		for (unsigned int j = 0; j < i;j++)
		{
			bool flag_1 = belong_or_not(structure_name, standby1[i]);
			bool flag_2 = belong_or_not(structure_name, standby1[j]);
			if (flag_1 == true && flag_2 == true)
			{
				standby[i][j] = get_distance_from_matrix(standby1[i], standby1[j]);
			}
			else
			{
				if (flag_1==true && flag_2==false)
				{
					standby[i][j] = get_distance_from_begin(standby1[i], standby1[j]);
				}
				if (flag_1==false && flag_2==true)
				{
					standby[i][j] = get_distance_from_begin(standby1[i], standby1[j]);
				}
			}
		}
	}

	matrix_len = matrix_len_renewal;	//replace old matrix_len
	delete []structure_name;
	structure_name = new string[matrix_len];
	for (unsigned int i = 0; i < matrix_len;i++)	//replace structure_name
	{
		structure_name[i] = standby1[i];
	}

	Deletearray(&distance_matrix,matrix_len+1);
	Newarray(&distance_matrix, matrix_len, matrix_len);
	
	for (unsigned int i = 0; i < matrix_len; i++)	//replace old distance matrix
	{
		for (unsigned int j = 0; j < matrix_len; j++)
		{
			distance_matrix[i][j] = standby[i][j];
		}
	}

	Deletearray(&standby,matrix_len);
	delete[]standby1;
}

void	//construct phylogenetic tree
UPGMA::construction_of_tree()
{
	for (unsigned int i = 0; i < matrix_len;i++)	//construct node
	{
		UPGMA_node a;
		a = create_node(structure_name[i],-1);
		node_of_tree.push_back(a);
	}

	int max_len = matrix_len;	

	for (unsigned int i = 0; i < max_len - 1;i++)
	{
		int column = min_distance().first;	//find the cloumn and row of minimum value in distance matrix
		int row = min_distance().second;
		string node_1 = structure_name[column];	//the name of node 1 which merged
		string node_2 = structure_name[row];	//the name of node 2 which merged
		double distance = distance_matrix[column][row];
		string new_node = node_1 + "+" + node_2;	//new node
		UPGMA_node node = create_node(new_node, -1);
		node_of_tree.push_back(node);
		for (unsigned int j = 0; j < node_of_tree.size();j++)	//construct new edge
		{
			if (get_node(node_of_tree[j])==node_1 || get_node(node_of_tree[j])==node_2)
			{
				change_level(node_of_tree[j], i + 1);
				UPGMA_edge edge = create_edge(node_of_tree[j], node, (distance / 2)-get_distance_to_bottom(get_node(node_of_tree[j])));
				edge_of_tree.push_back(edge);
			}
		}
		renewal_matrix(column, row); //renewal distance matrix
	}
	
	Deletearray(&distance_matrix_begin, max_len);
	Deletearray(&distance_matrix, matrix_len);
	delete [] structure_name;
	delete [] structure_name_begin;
}

vector<pair<string, string> > //construct vector which save aligment order
UPGMA::find_alignment_target()
{
	vector<pair<string,string> > f;
	string a,b;
	for (unsigned int i = 0; i < max_len - 1; i++)
	{
		for (unsigned int j = 0; j < node_of_tree.size(); j++)
		{
			if (get_node_level(node_of_tree[j]) == i + 1)
			{
				a = get_node(node_of_tree[j]);
				for (unsigned int m = j + 1; m < node_of_tree.size(); m++)
				{
					if (get_node_level(node_of_tree[m]) == i+1)
					{
						b = get_node(node_of_tree[m]);
						pair<string, string> aa(a, b);
						f.push_back(aa);
					}
				}
			}
		}
	}
	return f;
}