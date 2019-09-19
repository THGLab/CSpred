
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



#ifndef _BASIC_FUN_H
#define _BASIC_FUN_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string>
#include <string.h>
#include <malloc.h>

#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <algorithm>

#define getmax(a,b) a>b?a:b
#define getmin(a,b) a>b?b:a



using namespace std;




void PrintErrorAndQuit(char* sErrorString)
{
	cout << sErrorString << endl;
	exit(1);
}


template <class A> void NewArray(A *** array, int Narray1, int Narray2)
{
  *array=new A* [Narray1];
  for(int i=0; i<Narray1; i++) *(*array+i)=new A [Narray2];
};

template <class A> void DeleteArray(A *** array, int Narray)
{
  for(int i=0; i<Narray; i++)
    if(*(*array+i)) delete [] *(*array+i);
  if(Narray) delete [] (*array);
  (*array)=NULL;
};


char AAmap(string AA)
{
    char A=' ';
    if(     AA.compare("BCK")==0)   A='X';
    else if(AA.compare("GLY")==0)   A='G';
    else if(AA.compare("ALA")==0)   A='A';
    else if(AA.compare("SER")==0)   A='S';
    else if(AA.compare("CYS")==0)   A='C';
    else if(AA.compare("VAL")==0)   A='V';     
    else if(AA.compare("THR")==0)   A='T';
    else if(AA.compare("ILE")==0)   A='I';
    else if(AA.compare("PRO")==0)   A='P';
    else if(AA.compare("MET")==0)   A='M';
    else if(AA.compare("ASP")==0)   A='D';
    else if(AA.compare("ASN")==0)   A='N';
    else if(AA.compare("LEU")==0)   A='L';
    else if(AA.compare("LYS")==0)   A='K';
    else if(AA.compare("GLU")==0)   A='E';
    else if(AA.compare("GLN")==0)   A='Q';
    else if(AA.compare("ARG")==0)   A='R';
    else if(AA.compare("HIS")==0)   A='H';
    else if(AA.compare("PHE")==0)   A='F';
    else if(AA.compare("TYR")==0)   A='Y';
    else if(AA.compare("TRP")==0)   A='W';    
    else if(AA.compare("CYX")==0)   A='C';
    else
        A='Z'; //ligand
        
    return A;
}

void AAmap3(char A, char AA[3])
{
    if     ( A=='X')   strcpy(AA, "BCK");
	else if( A=='G')   strcpy(AA, "GLY");
	else if( A=='A')   strcpy(AA, "ALA");
	else if( A=='S')   strcpy(AA, "SER");
	else if( A=='C')   strcpy(AA, "CYS");
	else if( A=='V')   strcpy(AA, "VAL");
	else if( A=='T')   strcpy(AA, "THR");
	else if( A=='I')   strcpy(AA, "ILE");
	else if( A=='P')   strcpy(AA, "PRO");
	else if( A=='M')   strcpy(AA, "MET");
	else if( A=='D')   strcpy(AA, "ASP");
	else if( A=='N')   strcpy(AA, "ASN");
	else if( A=='L')   strcpy(AA, "LEU");
	else if( A=='K')   strcpy(AA, "LYS");
	else if( A=='E')   strcpy(AA, "GLU");
	else if( A=='Q')   strcpy(AA, "GLN");
	else if( A=='R')   strcpy(AA, "ARG");
	else if( A=='H')   strcpy(AA, "HIS");
	else if( A=='F')   strcpy(AA, "PHE");
	else if( A=='Y')   strcpy(AA, "TYR");
	else if( A=='W')   strcpy(AA, "TRP");
	else if( A=='C')   strcpy(AA, "CYX");
    else
        strcpy(AA, "UNK");           
}


void get_xyz(string line, double *x, double *y, double *z, char *resname, int *no)
{
    char cstr[50];    
    
    strcpy(cstr, (line.substr(30, 8)).c_str());
    sscanf(cstr, "%lf", x);
    
    strcpy(cstr, (line.substr(38, 8)).c_str());
    sscanf(cstr, "%lf", y);  
    
    strcpy(cstr, (line.substr(46, 8)).c_str());
    sscanf(cstr, "%lf", z);
    
    strcpy(cstr, (line.substr(17, 3)).c_str());
    *resname=AAmap(cstr);

    strcpy(cstr, (line.substr(22, 4)).c_str());
    sscanf(cstr, "%d", no);
}

int get_PDB_len(char *filename)
{
    int i=0;   
    string line;
    string atom ("ATOM "); 
    
    
    ifstream fin (filename);
    if (fin.is_open())
    {
        while ( fin.good() )
        {
            getline(fin, line);
            if(line.compare(0, atom.length(), atom)==0)
            {
                if( line.compare(12, 4, "CA  ")==0 ||\
                    line.compare(12, 4, " CA ")==0 ||\
                    line.compare(12, 4, "  CA")==0 )
                {
                    if( line.compare(16, 1, " ")==0 ||\
                        line.compare(16, 1, "A")==0 )
                    {                                  
                        i++;
                    }
                }                  
            }            
        }
        fin.close();
    }
    else
    {
		char message[5000];
		sprintf(message, "Can not open file: %s\n", filename);
        PrintErrorAndQuit(message);
    }
    
    return i;    
}

int get_PDB_len(char *filename, int& atomlen)
{
    int i = 0;
    string line;
    string atom("ATOM ");

    ifstream fin(filename);
    atomlen = 0;
    if (fin.is_open())
    {
        while (fin.good())
        {
            getline(fin, line);
            if (line.compare(0, atom.length(), atom) == 0)
            {
                atomlen++;

                if (line.compare(12, 4, "CA  ") == 0 || \
                    line.compare(12, 4, " CA ") == 0 || \
                    line.compare(12, 4, "  CA") == 0)
                {
                    if (line.compare(16, 1, " ") == 0 || \
                        line.compare(16, 1, "A") == 0)
                    {
                        i++;
                    }
                }
            }
        }
        fin.close();
    }
    else
    {
        char message[5000];
        sprintf(message, "Can not open file: %s\n", filename);
        PrintErrorAndQuit(message);
    }

    return i;
}


int read_PDB(char *filename, double **a, char *seq, int *resno)
{
    int i=0;
    string line, str;    
    string atom ("ATOM "); 
    
    
    ifstream fin (filename);
    if (fin.is_open())
    {
        while ( fin.good() )
        {
            getline(fin, line);
            if(line.compare(0, atom.length(), atom)==0)
            {
                if( line.compare(12, 4, "CA  ")==0 ||\
                    line.compare(12, 4, " CA ")==0 ||\
                    line.compare(12, 4, "  CA")==0 )
                {
                    if( line.compare(16, 1, " ")==0 ||\
                        line.compare(16, 1, "A")==0 )
                    {  
                        get_xyz(line, &a[i][0], &a[i][1], &a[i][2], &seq[i], &resno[i]);
                        i++;
                    }
                }                  
            }            
        }
        fin.close();
    }
    else
    {
		char message[5000];
		sprintf(message, "Can not open file: %s\n", filename);
        PrintErrorAndQuit(message);
    } 
    seq[i]='\0';   
    
    return i;
}

int get_ligand_len(char *filename)
{
    int i=0;
    char cstr[100];    
    string line;    
    string atom ("HETATM "); 
    string finish ("END"); 
    
    
    ifstream fin (filename);
    if (fin.is_open())
    {
        while ( fin.good() )
        {
            getline(fin, line);
            if(line.compare(0, atom.length(), atom)==0)
            {
                strcpy(cstr, (line.substr(12, 4)).c_str()); 
                   
                if(!strstr(cstr, "H"))
                {
                    if( line.compare(16, 1, " ")==0 ||\
                        line.compare(16, 1, "A")==0 )
                    {
                        i++;
                    }
                }                  
            }
            else if(line.compare(0, finish.length(), finish)==0) 
            {
                break;
            }          
        }
        fin.close();
    }
    else
    {
		char message[5000];
		sprintf(message, "Can not open file: %s\n", filename);
        PrintErrorAndQuit(message);
    } 
    
    return i;
}


int read_ligand(char *filename, double **a, char *seq, int *resno)
{
    int i=0;
    char cstr[100];    
    string line, str;    
    string atom ("HETATM "); 
    string finish ("END"); 
    
    
    ifstream fin (filename);
    if (fin.is_open())
    {
        while ( fin.good() )
        {
            getline(fin, line);
            if(line.compare(0, atom.length(), atom)==0)
            {
                strcpy(cstr, (line.substr(12, 4)).c_str()); 
                   
                if(!strstr(cstr, "H"))
                {
                    if( line.compare(16, 1, " ")==0 ||\
                        line.compare(16, 1, "A")==0 )
                    {                                  
                        get_xyz(line, &a[i][0], &a[i][1], &a[i][2], &seq[i], &resno[i]);
                        i++;
                    }
                }                  
            }
            else if(line.compare(0, finish.length(), finish)==0) 
            {
                break;
            }          
        }
        fin.close();
    }
    else
    {
		char message[5000];
		sprintf(message, "Can not open file: %s\n", filename);
        PrintErrorAndQuit(message);     
    } 
    seq[i]='\0';   
    
    return i;
}

int get_full_len(char *filename)
{
    int i = 0;
    string line;
    string atom("ATOM  ");
    string hetatm("HETATM");

    ifstream fin(filename);
    int atomlen = 0;
    if (fin.is_open())
    {
        while (fin.good())
        {
            getline(fin, line);
            if (line.compare(0, atom.length(), atom) == 0 || line.compare(0, hetatm.length(), hetatm) == 0)
            {
                atomlen++;
            }
        }
        fin.close();
    }
    else
    {
        char message[5000];
        sprintf(message, "Can not open file: %s\n", filename);
        PrintErrorAndQuit(message);
    }
    return atomlen;
}

int get_CONECT_len(char *filename)
{
    int i = 0;
    string line;
    string conect("CONECT");

    ifstream fin(filename);
    int connect_len = 0;
    if (fin.is_open())
    {
        while (fin.good())
        {
            getline(fin, line);
            if (line.compare(0, conect.length(), conect) == 0)
            {
                connect_len++;
            }
        }
        fin.close();
    }
    else
    {
        char message[5000];
        sprintf(message, "Can not open file: %s\n", filename);
        PrintErrorAndQuit(message);
    }
    return connect_len;
}

int read_CONECT(char *filename, string *conect_line)
{
    int i = 0;
    string line, str;
    string conect("CONECT");

    ifstream fin(filename);
    if (fin.is_open())
    {
        while (fin.good())
        {
            getline(fin, line);

            if (line.compare(0, conect.length(), conect) == 0)
            {
                conect_line[i] = line;
                i++;
            }
        }
        fin.close();
    }
    else
    {
        char message[5000];
        sprintf(message, "Can not open file: %s\n", filename);
        PrintErrorAndQuit(message);
    }
    return i;
}

void get_xyz_simple(string line, double *x, double *y, double *z)
{
    char cstr[50];    
    
    strcpy(cstr, (line.substr(30, 8)).c_str());
    sscanf(cstr, "%lf", x);
    
    strcpy(cstr, (line.substr(38, 8)).c_str());
    sscanf(cstr, "%lf", y);  
    
    strcpy(cstr, (line.substr(46, 8)).c_str());
    sscanf(cstr, "%lf", z);
}


void get_xyz_full(string line, char *fla, char *ato, char *res, char *num1, char *num2, double *x, double *y, double *z)
{
    char cstr[50];

    strcpy(fla, (line.substr(0, 6)).c_str());  //fla


    strcpy(num1, (line.substr(6, 5)).c_str());  //num1
    //sscanf(cstr, "%d", num1);

    strcpy(num2, (line.substr(22, 5)).c_str()); //num2
    //sscanf(cstr, "%d", num2);
    
    strcpy(ato, (line.substr(12, 4)).c_str());
    //sscanf(cstr, "%lf", ato);    

    strcpy(res, (line.substr(17, 3)).c_str());   //res
    //scanf(cstr, "%lf", res);
    
    strcpy(cstr, (line.substr(30, 8)).c_str());
    sscanf(cstr, "%lf", x);

    strcpy(cstr, (line.substr(38, 8)).c_str());
    sscanf(cstr, "%lf", y);

    strcpy(cstr, (line.substr(46, 8)).c_str());
    sscanf(cstr, "%lf", z);

}

int read_PDB_fullatom(char *filename, char **fla, char **ato, char **res, char **num1, char **num2, double **xyza)
{
    int i = 0;
    string line, str;
    string atom("ATOM  ");
    string hetatm("HETATM");

    ifstream fin(filename);
    if (fin.is_open())
    {
        while (fin.good())
        {
            getline(fin, line);

            if (line.compare(0, atom.length(), atom) == 0 || line.compare(0, hetatm.length(), hetatm) == 0)
            {
                get_xyz_full(line, fla[i], ato[i], res[i], num1[i], num2[i], &xyza[i][0], &xyza[i][1], &xyza[i][2]);                 
                i++;
            }
        }
        fin.close();
    }
    else
    {
        char message[5000];
        sprintf(message, "Can not open file: %s\n", filename);
        PrintErrorAndQuit(message);
    }
    return i;
}

int read_PDB_simple(char *filename, double **a)
{
    int i=0;
    string line, str;    
    string atom ("ATOM "); 
    
    
    ifstream fin (filename);
    if (fin.is_open())
    {
        while ( fin.good() )
        {
            getline(fin, line);
            if(line.compare(0, atom.length(), atom)==0)
            {
                if( line.compare(12, 4, "CA  ")==0 ||\
                    line.compare(12, 4, " CA ")==0 ||\
                    line.compare(12, 4, "  CA")==0 )
                {
                    if( line.compare(16, 1, " ")==0 ||\
                        line.compare(16, 1, "A")==0 )
                    {  
                        get_xyz_simple(line, &a[i][0], &a[i][1], &a[i][2]);
                        i++;
                    }
                }                  
            }            
        }
        fin.close();
    }
    else
    {
        char message[5000];
        sprintf(message, "Can not open file: %s\n", filename);
        PrintErrorAndQuit(message);
    } 
    
    return i;
}


double dist(double x[3], double y[3])
{
	double d1=x[0]-y[0];
	double d2=x[1]-y[1];
	double d3=x[2]-y[2];	
 
    return (d1*d1 + d2*d2 + d3*d3);
}

double dot(double *a, double *b)
{
  return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

void transform(double t[3], double u[3][3], double *x, double *x1)
{
    x1[0]=t[0]+dot(&u[0][0], x);
    x1[1]=t[1]+dot(&u[1][0], x);
    x1[2]=t[2]+dot(&u[2][0], x);
}

void do_rotation(double **x, double **x1, int len, double t[3], double u[3][3])
{
    for(int i=0; i<len; i++)
    {
        transform(t, u, &x[i][0], &x1[i][0]);
    }    
}



void output_align1(int *invmap0, int len)
{
	for(int i=0; i<len; i++)
	{
		if(invmap0[i]>=0)
		{
			cout << invmap0[i]+1 << " ";
		}			
		else
			cout << invmap0[i] << " ";

	}	
	cout << endl << endl;	
}


int output_align(int *invmap0, int len)
{
	int n_ali=0;
	for(int i=0; i<len; i++)
	{		
		cout <<  invmap0[i] << " ";
		n_ali++;		
	}	
	cout << endl << endl;	

	return n_ali;
}

#endif
