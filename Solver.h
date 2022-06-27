/* 
 * File:   Solver.h
 * Author: koosha
 *
 * Created on November 26, 2011, 4:55 PM
 */

#ifndef SOLVER_H
#define	SOLVER_H
#include "POMDP.h"

#define MIN     -10
#define INITIAL_VALUE 0
#define NUM_ITERATIONS  151

class Solver {
public:
    static void MDP () ;
    static void MDP_det () ;
    static void SLAP_pairs (float difference_threshold) ; 
    static float online_planner (float compare_ratio);
    static int act_and_observe (int &real_state, int action);
    static void update_bel ( int action , int observation) ;
    static int choose_start_state () ;
    static void get_data(char* file_name) ;
    static void get_data() ;
    static void update_bel_action (int action) ;
    static int find_string (char** list , int list_size , char* key );
    static void homecare_data();
    static float MLS();
    static float QMDP();
    Solver();
    ~Solver();

};

void homecare_date ();

#endif	/* SOLVER_H */

