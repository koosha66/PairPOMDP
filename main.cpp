/* 
 * File:   main.cpp
 * Author: koosha
 *
 * Created on November 25, 2011, 4:08 PM
 */

#include <cstdlib>
#include <stdio.h>
#include "Solver.h"
#include <iostream>
using namespace std;
/*
 * 
 */
int main(int argc, char** argv) {
//POMDP P = POMDP("tagAvoid.txt") ;
Solver::get_data("hallway2.txt") ;
//Solver::homecare_data() ;
Solver::SLAP_pairs(.80);
cout << "end" << endl;
cout.flush();
Solver::MDP();
int i,j ;
    for (j=0 ; j < 10;j++)
    {
        float sum_reward = 0 ; 
        for (i=0; i < 10000; i++)
        {
            float temp_reward = Solver::online_planner(19);
            //printf ("%f " , temp_reward);
            sum_reward = sum_reward + temp_reward ;
        }
        cout << "mean=" << sum_reward/i << endl ;
        cout.flush();
    }
    return 0 ;
    
}
