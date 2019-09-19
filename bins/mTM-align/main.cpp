#include "TMM.h"

#include <ctime>
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdlib.h>
int main(int argc, char *argv[])
{
	double start, finish;
	start = clock();

	string version = "20180725";


	TMM T;
	if (argc < 2) 
    {
        T.print_help(argv[0]);        
    } 
	bool i_opt, v_opt, h_opt, o_opt, p_opt, t_opt;
	i_opt = v_opt = h_opt = o_opt = p_opt = t_opt = false;
	char *input_list, *out_name;
	string tmp1 = " ";
	string tmp2 = "result.pdb";
	int len1 = tmp1.length();
	int len2 = tmp2.length();
	input_list = new char [len1+1];
	out_name = new char [len2+1];
	strcpy(input_list, tmp1.c_str());
	strcpy(out_name, tmp2.c_str());
    for(int i = 1; i < argc; i++)
	{
		if ( !strcmp(argv[i],"-i") && i < (argc-1) ) 
		{
			input_list = argv[i + 1];      i_opt = true; i++;
		}

		else if (!strcmp(argv[i], "-o") && i < (argc-1) ) 
		{
			out_name = argv[i + 1];      o_opt = true; i++;
		}

		else if ( !strcmp(argv[i],"-v") ) 
		{
			v_opt = true; 
		}
		else if ( !strcmp(argv[i],"-h") ) 
		{ 
			h_opt = true; 
		}
	}

	if(h_opt)
	{
		T.print_help(argv[0]);    			
	}
		
	if(v_opt)
	{
		cout <<endl;
		cout <<endl;
		cout << " ***************************************************************" << endl;
		cout << " *                 mTM-align (Version "<< version <<")                *" << endl;
		cout << " * An algorithm for multiple protein structure alignment (MSTA)*"<<endl;
		cout << " * Reference: Dong, et al, Bioinformatics, 34: 1719-1725 (2018)*"<<endl;
		cout << " * Please email your comments to: yangjy@nankai.edu.cn         *"<<endl;
		cout << " ***************************************************************" << endl;
		cout <<endl;
		exit(EXIT_FAILURE);
	}

	if(!i_opt)
	{
		cout << "Please provide option -i and inputlist!" << endl;
		exit(EXIT_FAILURE);
	}

	if(i_opt)
	{
		if(!strcmp(input_list, " "))
		{
			cout << "Please provide inputlist!" << endl;
			exit(EXIT_FAILURE);
		}
	}

	ifstream fin(input_list);
	const int Line_length = 500;
	char str[Line_length];
	//char *str = new char[Line_length];
	vector<char*> input_vector;
	while (fin.getline(str, Line_length))
	{
		char *tmp = new char [Line_length];
		strcpy(tmp, str);
		input_vector.push_back(tmp);
	}

	T.matrix_output(input_vector);	
	T.programming(input_vector, out_name);


	cout << "The alignment in fasta format: result.fasta"<<endl;
	cout << "The superimposed structures in PDB format:  "<< out_name<<endl;
	cout << "The superimposed structures in common core region in PDB format:  cc.pdb"<<endl;
	cout << "The pairwise TM-score:  pairwise_TMscore.txt"<<endl;
	cout << "The pairwise RMSD:  pairwise_rmsd.txt"<<endl;
	cout << "The distance matrix: infile"<<endl;
	cout << "The pairwise superimposed structures: *_pair.pdb"<<endl;
	remove("matrix.txt");
	//finish = clock();
	//cout<<"Total running time is: "<<(double)(finish-start)/CLOCKS_PER_SEC<<" seconds"<<endl;
	return 0;
}

