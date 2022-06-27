/* 
 * File:   Solver.cpp
 * Author: koosha
 * 
 * Created on November 26, 2011, 4:55 PM
 */

#include "Solver.h"
#include <vector>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
using namespace std ;

float pair_values[5500][5500] ;
float pair_values2[5500][5500];
//int pair_actions2 [5500][5500];
float MDP_values_actions[12000] ;
int pair_actions [5500][5500] ;
int list[2][7000000] ; 
float belief[5500] ;
float temp[5500] ;
int list_states[5500] ;
float backup_belief [5500] ;
FILE* in ;

int nrStates ;
int nrActions ;
int nrObservations ;
float gamma ;
float reward [550][550][10] ;
float transition [550][550][10] ;
float observation [550][10][1000] ;
float start [550] ;
int states_actions [550][10] ;
int states_observations [550][10] ;
vector<int> neighbors [550][10];


Solver::Solver() {
}
Solver::~Solver() {
}
void Solver::MDP(){
     // the first n cells are values and the next n are best actions
    int iteration , i, j , k  ; 
    float max_value,value ; 
    
    for (i=0;i<2* nrStates;i++)
        MDP_values_actions[i] = 0 ;
    
    bool end_iter =false;
    iteration =0 ;
    while (iteration<1000)
    {
        iteration++;
        end_iter =true;
        for (i=0 ; i< nrStates; i++)
        { 
            float pre_val = MDP_values_actions[i];
            max_value = MIN ;
            for (j=0 ; j< nrActions ; j++)
            {
                value = 0 ; 
                for (k=0 ; k< neighbors[i][j].size(); k++)
                    value = value + MDP_values_actions[neighbors[i][j][k]]* transition[neighbors[i][j][k]][i][j] ;
                value = (value* gamma)+  reward[i][i][j] ;
               // printf ("ac=%d,val=%f " , j , value);
                
                if (value > max_value)
                {
                    max_value = value ;
                    MDP_values_actions[i] = value ; 
                    MDP_values_actions[i+ nrStates] = j ;
                }               
            }
            if (MDP_values_actions[i]!=pre_val)
                end_iter = false;
        }
    }
    //cout <<iteration<<endl;
    FILE* ac = fopen ("MDPAc.txt", "w") ;
    FILE* val = fopen ("MDPVal.txt", "w");
    for (i=0 ; i <nrStates;i++)
    {
        fprintf (ac, "%f ", MDP_values_actions[i+ nrStates]);
        fprintf (val, "%f ", MDP_values_actions[i]);
    }
}
void Solver::MDP_det(){   
    //float *MDP_values_actions = new float[2* nrStates] ; // the first n cells are values and the next n are best actions
    int iteration , i, j ; 
    float max_value,value ; 
    
    for (i=0;i<2* nrStates;i++)
        MDP_values_actions[i] = 0 ;
    
    for (iteration =0; iteration < 100 ; iteration++)
    {
        for (i=0 ; i< nrStates; i++)
        { 
            max_value = MIN ;
            for (j=0 ; j< nrActions ; j++)
            { 
                value = (MDP_values_actions[ states_actions[i][j]]* gamma)+  reward[i][i][j] ;   
                if (value > max_value)
                {
                    max_value = value ;
                    MDP_values_actions[i] = value ; 
                    MDP_values_actions[i+ nrStates] = j ;
                }               
            }
        }
    }
}
void Solver::SLAP_pairs(float difference_threshold){
    int i,j,ac,l,iter,number_states ;
    float max_value, temp_value ;
    bool is_different, change ;
    float** one_action_localization = new float*[5500];
    for (i=0;i<5500;i++)
        one_action_localization[i]= new float[5500]; 
    double b = clock();
    Solver::MDP();
    /*
    /////////////////////////////////////////////////   high_low: 
    number_states = 0 ; 
    max_value = MIN;
    change = false ;
    //initialization:
    for (i=0 ; i< nrStates ; i++)
        for (j=0 ; j< nrStates ; j++)
        {
            pair_actions[i][j] = -1 ;
            pair_values2[i][j] = INITIAL_VALUE ;
            one_action_localization[i][j] = MIN ;
        }
    for (i=0; i<  nrStates ; i++)
        for (j=0 ; j<nrStates ; j++)
        {
            if (i==j)
                continue;
            change = false ;
            max_value = MIN ;
            for (ac =0 ; ac <  nrActions ; ac++)                    
                {
                    float cur_dif = 0 ;
                    for (int ii= 0 ; ii< neighbors[i][ac].size() ; ii++)
                        for (int jj= 0 ; jj < neighbors[j][ac].size();jj++)
                        {
                            int s1 = neighbors[i][ac][ii];
                            int o1 = states_observations[s1][ac];
                            int s2 = neighbors[j][ac][jj] ;
                            int o2 = states_observations[s2][ac];
                            cur_dif += transition[s1][i][ac]*transition[s2][j][ac]*(observation[s1][ac][o1]*(1- observation[s2][ac][o1]) +  observation[s2][ac][o2]*(1- observation[s1][ac][o2]));
                        }   
                    if (cur_dif >= 2*difference_threshold)               
                       {
                          temp_value = .75*reward[states_actions[i][ac]][i][ac]+ .25*reward[ states_actions[j][ac]][j][ac]+ gamma*(.75*MDP_values_actions[ states_actions[i][ac]]+.25*MDP_values_actions[ states_actions[j][ac]]); 
                            if (temp_value>max_value)
                           {
                                max_value = temp_value ;
                                pair_values2[i][j] = temp_value ;
                                pair_actions[i][j] = ac ; 
                                one_action_localization[i][j] = temp_value ;
                                change = true ;
                           }
                        }
                }
            
            if (change == false )
            {
                list[0][number_states] = i ;
                list[1][number_states] = j ;
                number_states++ ;
            }  
        }   
    cout <<number_states <<endl;
    cout.flush();
    for (iter=0;iter<200;iter++) {  
        for (l=0;l<number_states;l++)
        {
          i=list[0][l];
          j=list[1][l];
         max_value = one_action_localization[i][j] ;
          for (ac=0 ; ac < nrActions; ac++)
          {
              temp_value = 0;
            for (int ni=0 ; ni< neighbors[i][ac].size() ; ni++)
                for (int nj=0 ; nj< neighbors[j][ac].size() ; nj++)
                    temp_value+= transition[neighbors[i][ac][ni]][i][ac]*transition[neighbors[j][ac][nj]][j][ac]*(.75*reward[neighbors[i][ac][ni]][i][ac]+ .25*reward[neighbors[j][ac][nj]][j][ac]+ gamma*pair_values[neighbors[i][ac][ni]][neighbors[j][ac][nj]]);
            
          //   temp_value = .75*reward[states_actions[i][ac]][i][ac]+ .25*reward[states_actions[j][ac]][j][ac]+ gamma*pair_values2[states_actions[i][ac]][states_actions[j][ac]];         
              if (temp_value>max_value)
                        {
                            max_value = temp_value ;
                            pair_values2[i][j] = temp_value ;
                            pair_actions[i][j] = ac ;
                        }
          }         
        } 
    }
     */ 
    cout <<"begin" <<endl;
    cout.flush();
    
    /////////////////////////////////////////////////  equal:
    number_states = 0 ; 
    max_value = MIN;
    change = false ;
    //initialization:
    for (i=0 ; i< nrStates ; i++)
        for (j=0 ; j< nrStates ; j++)
        {
            pair_actions[i][j] = -1 ;
            pair_values[i][j] = INITIAL_VALUE ;
            one_action_localization[i][j] = MIN ;
        }
    for (i=0; i<  nrStates ; i++)
        for (j=0 ; j<i ; j++)       
        {
            change = false ;
            max_value = MIN ;
            for (ac =0 ; ac <  nrActions ; ac++)                    
                {
                    float cur_dif = 0 ;
                    for (int ii= 0 ; ii< neighbors[i][ac].size() ; ii++)
                        for (int jj= 0 ; jj < neighbors[j][ac].size();jj++)
                        {
                            int s1 = neighbors[i][ac][ii];
                            int o1 = states_observations[s1][ac];
                            int s2 = neighbors[j][ac][jj] ;
                            int o2 = states_observations[s2][ac];
                            cur_dif += transition[s1][i][ac]*transition[s2][j][ac]*(observation[s1][ac][o1]*(1- observation[s2][ac][o1]) +  observation[s2][ac][o2]*(1- observation[s1][ac][o2]));
                        }   
                    if (cur_dif >= 2*difference_threshold)               
                       {
                          temp_value = .5*( reward[states_actions[i][ac]][i][ac]+ reward[ states_actions[j][ac]][j][ac])+ gamma*.5*(MDP_values_actions[ states_actions[i][ac]]+MDP_values_actions[ states_actions[j][ac]]); 
                            if (temp_value>max_value)
                           {
                                max_value = temp_value ;
                                pair_values[i][j] = pair_values[j][i] = temp_value ;
                                pair_actions[i][j] = pair_actions[j][i] = ac ; 
                                one_action_localization[i][j] = one_action_localization[j][i] = temp_value ;
                                change = true ;
                           }
                        }
                }
            
            if (change == false )
            {
                list[0][number_states] = i ;
                list[1][number_states] = j ;
                number_states++ ;
            }  
        }   
    int s1,s2 ;
    bool end_iter=false;
    iter = 0;
    for( iter=0;iter<1000;iter++) {   
        //end_iter = true;
       //iter++;
        for (l=0;l<number_states;l++)
        {
         i=list[0][l];
         j=list[1][l];
        float pre_value = pair_values[i][j];
         max_value = one_action_localization[i][j] ;
          for (ac=0 ; ac < nrActions; ac++)
          {
              temp_value = 0;
           // for (int ni=0 ; ni< neighbors[i][ac].size() ; ni++)
             //   for (int nj=0 ; nj< neighbors[j][ac].size() ; nj++)
               //     temp_value+= transition[neighbors[i][ac][ni]][i][ac]*transition[neighbors[j][ac][nj]][j][ac]*(.5*(reward[neighbors[i][ac][ni]][i][ac]+ reward[neighbors[j][ac][nj]][j][ac])+ gamma*pair_values[neighbors[i][ac][ni]][neighbors[j][ac][nj]]);
             temp_value = (.5*(reward[states_actions[i][ac]][i][ac]+ reward[states_actions[j][ac]][j][ac])+ gamma*pair_values[states_actions[i][ac]][states_actions[j][ac]]);         
             if (temp_value>max_value)
                        {
                            max_value = temp_value ;
                            pair_values[i][j] = pair_values[j][i] = temp_value ;
                            pair_actions[i][j] = pair_actions[j][i] = ac ;
                        }
          }  
        // if (pair_values[i][j] - pre_value > 0.05 || pair_values[i][j] - pre_value < -0.05)
          // end_iter = false;
        } 
        // cout << iter << endl;
        // cout.flush();
    }
    cout << iter << endl;
    double e = clock();
    cout << (e-b)/CLOCKS_PER_SEC <<endl;
    FILE* out1 = fopen ("pair_values.txt", "w") ;
    FILE* out2 = fopen ("pair_actions.txt","w") ;
    for (i=0;i< nrStates;i++)
    {
        for (j=0;j< nrStates;j++)
        {
                fprintf (out1, "%f ", pair_values[i][j]);
                fprintf (out2, "%d ", pair_actions[i][j]);
        }
        fprintf(out1,"\n");
        fprintf(out2,"\n");
    }  
    for (i=0;i<5500;i++)
        delete[]one_action_localization[i];
    delete[]one_action_localization;
}
float Solver::online_planner(float compare_ratio){
    //int end = 0 ; 
    //int old_real_state ;
    int i , j, iter , ac ,s1,s2 ;
    int action,o;
    int max_bel, number_states, old_number_states ;
    float max_utility, utility,total_reward ;
    max_utility = MIN ;
    bool* valid_action = new bool[nrActions];
    //Solver::MDP();
    int real_state;
    int real_states [NUM_ITERATIONS] ;
    int actions [NUM_ITERATIONS] ;
    int num_iter = 0 ;
    srand ( time(NULL) );
    for (i=0 ;i< nrStates;i++)
    {
        belief[i]=  start[i];
    }
        real_state = choose_start_state () ;
        //cout << "real= " << real_state << " " ;
        //cout.flush();
        
        
    double b = clock();
    for (iter = 0 ; iter < NUM_ITERATIONS ; iter++)
    { 
    //    if (iter > 0) 
      //      if (real_states[iter-1] > nrStates-5)
        //        break ;
        num_iter++ ;
        real_states[iter] = real_state ;
        
        
        for (i=0;i< nrActions;i++)
            valid_action[i] = false ;
        max_bel = 0 ;
        for (i=1; i< nrStates; i++)
            if (belief[i]> belief[max_bel])
                max_bel = i ;
        
        
        number_states=0 ;
        for (i=0; i< nrStates; i++)
            if (belief[i]>belief[max_bel]/compare_ratio)
            {
                list_states[number_states++]=i;
            }    
       
        if (number_states ==1)
        {
            action = (int) MDP_values_actions[list_states[0]+ nrStates] ;
        }
        else
        {
            
            max_utility = MIN ;
            action = 0 ;
            for (i=0;i<number_states;i++)
                for (j=0;j<i;j++)
                    if (pair_actions[list_states[i]][list_states[j]]>= 0)
                        valid_action[pair_actions[list_states[i]][list_states[j]]] = true ;
                
            for (ac = 0 ; ac <  nrActions; ac++)
              if (valid_action[ac]==true)
               {                                    
                   utility = 0 ;            
                   for (i=0 ; i<number_states ; i++)
                            for (j=0 ; j < i ; j++)
                            {
                               s1 = list_states[i];
                               s2 = list_states[j] ;                         
                               /*
                               if (belief[s1]<belief[s2]) {
                                   int temp_s = s1;
                                   s1=s2;
                                   s2= temp_s;
                               }*/
           //for (int n1=0 ; n1< neighbors[s1][ac].size() ; n1++)
                //for (int n2=0 ; n2< neighbors[s2][ac].size() ; n2++)
                 // utility+= belief[s1]*belief[s2]*transition[neighbors[s1][ac][n1]][s1][ac]*transition[neighbors[s2][ac][n2]][s2][ac]*(.5*(reward[neighbors[s1][ac][n1]][s1][ac]+ reward[neighbors[s2][ac][n2]][s2][ac])+ gamma*pair_values[neighbors[s1][ac][n1]][neighbors[s2][ac][n2]]);
                               
                               
                            //  utility = utility + belief[s1]*reward[states_actions[s1][ac]][s1][ac]+belief[s2]*(reward[states_actions[s2][ac]][s2][ac]);
                             // if (belief[s1]/belief[s2]<= 2) {
                              utility = utility + belief[s1]*belief[s2]*(.5*reward[states_actions[s1][ac]][s1][ac]+.5*reward[states_actions[s2][ac]][s2][ac]+gamma*pair_values[states_actions[s1][ac]][states_actions[s2][ac]]) ;
                              // } else {
                                 // utility = utility + belief[s1]*belief[s2]*(.75*reward[states_actions[s1][ac]][s1][ac]+.25*reward[states_actions[s2][ac]][s2][ac]);
                                 // utility = utility + gamma*belief[s1]*belief[s2]*pair_values2[states_actions[s1][ac]][states_actions[s2][ac]] ; 
                              // }
                              }
                   /*
                   update_bel_action(ac);
                   int number_states2=0 ;
                   int list_states2[5500];
                   int  max_bel2 = 0 ;
                   for (i=1; i< nrStates; i++)
                        if (temp[i]> temp[max_bel2])
                                max_bel2 = i ;
                   for (i=0; i< nrStates; i++)
                        if (temp[i]>temp[max_bel]/compare_ratio)
                                list_states2[number_states2++]=i;
                   
                   for (i=0 ; i<number_states ; i++)
                            for (j=0 ; j < i ; j++)
                            {
                               s1 = list_states2[i];
                               s2 = list_states2[j] ;
                               utility = utility + gamma*temp[s1]*temp[s2]*pair_values[s1][s2] ;
                            }
                   */
                   if (utility>max_utility)
                   {
                        max_utility = utility ;
                        action = ac ;
                   }
                }
        }
         o = Solver::act_and_observe (real_state, action);       
         actions[iter] = action ;
         Solver::update_bel (action,o) ;
        // printf ("%d " , action ) ;
    }
    total_reward = 0 ; 
    float discount = 1 ;
    for (i=0 ; i < num_iter-1 ; i++)
    {
        discount = discount * gamma ;
        total_reward = total_reward + discount*reward[real_states[i+1]][real_states[i]][actions[i]] ;
    }
    double e = clock();
   //printf ("time= %f ",(e-b)/CLOCKS_PER_SEC);
   //printf ("last_state=%d  ", real_states[num_iter]);
    return total_reward ;
}

int Solver::act_and_observe (int& real_state,int action){
    //srand ( time(NULL) );
    float r = rand()% 32000 ;
    r =r / 32000 ;
    int i ; 
    float sum = 0 ; 
    for (i=0 ; i <  nrStates-1 ; i++)
    {
        sum = sum +  transition[i][real_state][action] ;
        if ( sum > r )
            break ;
    }
    real_state = i ;
    sum = 0 ;
    r = rand()% 32000 ;
    r =r / 32000 ;
    for (i=0 ; i <  nrObservations-1 ; i++)
    {
        sum = sum +  observation[real_state][action][i] ;
        if ( sum > r )
            return i ;
    }    
    return i ;
}


void Solver::update_bel (int action, int o){
    int i , j ;
    float sum_belief = 0 ;
    for (i=0 ; i<  nrStates; i++ )
        temp[i]= 0 ;
    for (i=0 ; i <  nrStates ; i++)
        for (j=0 ; j< neighbors[i][action].size() ; j++)
            temp[neighbors[i][action][j]] = temp[neighbors[i][action][j]] + belief[i]* transition[neighbors[i][action][j]][i][action] ;

    for (i=0 ; i< nrStates ; i++)
    {
        temp[i] = temp[i]* observation[i][action][o] ;
        sum_belief = sum_belief + temp[i] ;
    }
    //printf ("sum=%f" , sum_belief);
    for (i=0 ; i <  nrStates ; i++ )
        belief[i]= temp[i] / sum_belief ;
}

void Solver::update_bel_action (int action)
{
    int i , j ;
    for (i=0 ; i<  nrStates; i++ )
        temp[i]= 0 ;
    for (i=0 ; i <  nrStates ; i++)
        for (j=0 ; j< neighbors[i][action].size() ; j++)
            temp[neighbors[i][action][j]] = temp[neighbors[i][action][j]] + belief[i]* transition[neighbors[i][action][j]][i][action] ;
    
    /*for (i=0; i< nrObservations ; i++)
    {
        pr_observations[i]= 0 ;
        for (j=0 ; j<nrStates ; j++)
            pr_observations[i] = pr_observations[i] + observation[j][action][i]*temp[j] ;
    }*/
    
}

int Solver::choose_start_state(){
    float r = rand()% 32000 ;
    r =r / 32000 ;
    int i ; 
    float sum = 0 ; 
    for (i=0 ; i <  nrStates-1 ; i++)
    {
        sum = sum +  start[i];
        if ( sum > r )
            return i ;
    }
    return i ;
}
void Solver::get_data(char* file_name){
     int i, j,k ; 
    float max_transition , max_observation ;
    FILE* in = fopen(file_name , "r") ;
    fscanf (in,"%d",&nrStates) ;
    fscanf (in,"%d",&nrActions) ;
    fscanf (in,"%d",&nrObservations) ;
    fscanf (in,"%f",&gamma) ;
    for (i=0 ; i<nrActions ; i++)
        for (j=0 ; j<nrStates ; j++)
            for (k=0 ; k<nrStates ; k++)
                fscanf (in,"%f",&reward[j][k][i]);
    for (i=0 ; i<nrObservations ; i++)
        for (j=0 ; j<nrStates ; j++)
            for (k=0 ; k<nrActions ; k++)
                fscanf (in,"%f",&observation[j][k][i]);
        
    for (i=0 ; i<nrActions ; i++)
        for (j=0 ; j<nrStates ; j++)
            for (k=0 ; k<nrStates ; k++)
                fscanf (in,"%f",&transition[j][k][i]);
    
    for (i=0;i<nrStates ; i++)
        fscanf (in,"%f",&start[i]) ;   
    
    for (i=0 ;i < nrStates ; i++)
        for (j=0 ; j<nrActions ; j++)
        {
            max_transition = transition[0][i][j] ;
            states_actions[i][j] = 0 ;
            for (k=1 ; k<nrStates ; k++)
                if (transition[k][i][j] > max_transition)
                {
                        max_transition = transition[k][i][j] ;
                        states_actions[i][j] = k ;
                }       
        }
    

    for (i=0 ;i < nrStates ; i++)
        for (j=0 ; j<nrActions ; j++)
        {
            max_observation = observation[i][j][0] ;
            states_observations[i][j] = 0 ;
            for (k=1 ; k<nrObservations ; k++)
                if (observation[i][j][k] > max_observation)
                {
                        max_observation = observation[i][j][k] ;
                        states_observations[i][j] = k ;
                }       
        }
    for (i=0;i<nrStates;i++)
        for (j=0 ; j<nrStates ; j++)
            for (k=0 ; k<nrActions ; k++)
                if (transition[j][i][k]>0)
                    neighbors[i][k].push_back(j);
     
     // for ( i = 0 ; i <nrStates ; i++)       
      //    for (j=0 ; j<neighbors[i][0].size(); j++)
       //       printf ("%f ", transition[neighbors[i][0][j]][i][0]);
            
}


void Solver::homecare_data(){
    int i, j,k ; 
    char temp[256];
    in = fopen("homecare.pomdp" , "r") ;
    fscanf (in,"%d",&nrStates) ;
    fscanf (in,"%d",&nrObservations) ;
    fscanf (in,"%d",&nrActions) ;
    fscanf (in,"%f",&gamma) ;
    char** list_states = new char* [nrStates];
    for (i=0;i<nrStates ; i++)
        list_states[i] = new char [40];
    char** list_observations = new char* [nrObservations];
    for (i=0;i<nrObservations ; i++)
        list_observations[i] = new char [40];
    char** list_actions = new char* [nrActions];
    for (i=0;i<nrActions ; i++)
        list_actions[i] = new char [40];
    

    
    for (i=0 ; i<nrActions ; i++)
        for (j=0 ; j<nrStates ; j++)
            for (k=0 ; k<nrStates;k++)
            {
                reward[j][k][i]=0;
                transition[j][k][i]= 0;
            }
    
    for (i=0 ; i<nrObservations ; i++)
        for (j=0 ; j<nrStates ; j++)
            for (k=0 ; k<nrActions ; k++)
                observation[j][k][i]=0;
    
   for (i=0 ; i<nrActions ; i++)
       fscanf (in, "%s", list_actions[i]);
   for (i=0 ; i<nrObservations ; i++)
       fscanf (in, "%s", list_observations[i]);
   for (i=0 ; i<nrStates ; i++)
       fscanf (in, "%s", list_states[i]);

     
       
   for (i=0;i<nrStates ; i++)
        fscanf (in,"%f",&start[i]) ; 
    
   
    int a1,s1,s2,o1;
    fscanf (in , "%s",temp) ;
    while (!feof(in))
    {
        fscanf (in , "%s",temp) ;
        if (temp[0]=='T')
        {
            fscanf(in,"%s", temp);
            a1 = find_string(list_actions,nrActions,temp);
            fscanf(in,"%s", temp);
            fscanf(in,"%s", temp);
            s1 = find_string(list_states,nrStates,temp);
            fscanf(in,"%s", temp);
            fscanf(in,"%s", temp);
            s2 = find_string(list_states,nrStates,temp);
            fscanf(in,"%f", &transition[s2][s1][a1]);
        }
        if (temp[0]=='O')
        {
            fscanf(in,"%s", temp);
            a1 = find_string(list_actions,nrActions,temp);
            fscanf(in,"%s", temp);
            fscanf(in,"%s", temp);
            s1 = find_string(list_states,nrStates,temp);
            fscanf(in,"%s", temp);
            fscanf(in,"%s", temp);
            o1 = find_string(list_observations,nrObservations,temp);
            fscanf(in,"%f", &observation[s1][a1][o1]);
        }
        
        if (temp[0]=='R')
        {
            fscanf(in,"%s", temp);
            a1 = find_string(list_actions,nrActions,temp);
            fscanf(in,"%s", temp);   //:
            fscanf(in,"%s", temp);   //*
            fscanf(in,"%s", temp);   //:
            fscanf(in,"%s", temp);
            s1 = find_string(list_states,nrStates,temp);
            fscanf(in,"%s", temp); // :
            fscanf(in,"%s", temp);  //*
            float temp_r ;
            fscanf(in,"%f", &temp_r);
            for (i=0 ; i<nrStates ;i++)
                reward [s1][i][a1] = temp_r ; 
        }
        if (temp[0] == 'E')
            break ;
    }
    Solver::get_data();
    
    /*
    cout << find_string(list_actions, nrActions , "w")<< endl  ;
    cout << find_string(list_states,nrStates,"Srv10rh5tv10th5c1")<<endl ;;
    cout << find_string(list_states,nrStates,"Srv10rh4tv9th5c0")<<endl ;
    cout << find_string(list_observations,nrObservations,"Orv10rh5tv10th5c1")<<endl ;
    cout << transition[5290][5407][3] <<endl ;
    cout << observation[5407][3][926] <<endl ;
    cout << reward [5407][1][3] << endl ;
     */ 
}

int Solver::find_string (char** list , int list_size , char* key )
{
    int i ; 
    for (i=0 ; i <list_size ; i++)
    {
        //cout << list[i] << " " << key << endl ;
        if (strcmp(list[i],key)==0)                
            return i ;
    }
    cout << "ERR " << key << " " << list_size<< endl ;
    cout.flush();
    return -1 ;
}

void Solver::get_data(){
     int i, j,k ; 
    float max_transition , max_observation ;
    for (i=0 ;i < nrStates ; i++)
        for (j=0 ; j<nrActions ; j++)
        {
            max_transition = transition[0][i][j] ;
            states_actions[i][j] = 0 ;
            for (k=1 ; k<nrStates ; k++)
                if (transition[k][i][j] > max_transition)
                {
                        max_transition = transition[k][i][j] ;
                        states_actions[i][j] = k ;
                }       
        }
    

    for (i=0 ;i < nrStates ; i++)
        for (j=0 ; j<nrActions ; j++)
        {
            max_observation = observation[i][j][0] ;
            states_observations[i][j] = 0 ;
            for (k=1 ; k<nrObservations ; k++)
                if (observation[i][j][k] > max_observation)
                {
                        max_observation = observation[i][j][k] ;
                        states_observations[i][j] = k ;
                }       
        }
    for (i=0;i<nrStates;i++)
        for (j=0 ; j<nrStates ; j++)
            for (k=0 ; k<nrActions ; k++)
                if (transition[j][i][k]>0)
                    neighbors[i][k].push_back(j);

}

float Solver::MLS(){
    
    int i ,iter ;
    int action,o;
    int max_bel, number_states;
    float total_reward ;
    int real_state;
    int real_states [NUM_ITERATIONS] ;
    int actions [NUM_ITERATIONS] ;
    int num_iter = 0 ;
    srand ( time(NULL) );
    for (i=0 ;i< nrStates;i++)
        belief[i]=  start[i];
    real_state = choose_start_state () ;
    //cout << "real=" << real_state <<endl;
    //cout.flush();
    double b = clock();
    for (iter = 0 ; iter < NUM_ITERATIONS ; iter++)
    { 
        num_iter++ ;
        real_states[iter] = real_state ; 
        max_bel = 0 ;
        for (i=1; i< nrStates; i++)
            if (belief[i]> belief[max_bel])
                max_bel = i ;
        number_states=0;    
        for (i=0; i< nrStates; i++)            
            if (belief[i]==belief[max_bel])
                list_states[number_states++]=i;
        
        int mls = rand()% number_states ;
        action = (int) MDP_values_actions[mls+ nrStates] ;
        
         o = Solver::act_and_observe (real_state, action);       
         actions[iter] = action ;
         Solver::update_bel (action,o) ;
        // printf ("%d " , action ) ;
    }
    total_reward = 0 ; 
    float discount = 1 ;
    for (i=0 ; i < num_iter-1 ; i++)
    {
        discount = discount * gamma ;
        total_reward = total_reward + discount*reward[real_states[i+1]][real_states[i]][actions[i]] ;
    }
    double e = clock();
   //printf ("time= %f ",(e-b)/CLOCKS_PER_SEC);
   //printf ("last_state=%d  ", real_states[num_iter]);
    return total_reward ;
}


float Solver::QMDP(){
    //int end = 0 ; 
    //int old_real_state ;
    int i , j, iter , ac ,s1,s2 ;
    int action,o;
    int max_bel, number_states, old_number_states ;
    float max_utility, utility,total_reward ;
    max_utility = MIN ;
    bool* valid_action = new bool[nrActions];
    //Solver::MDP();
    int real_state;
    int real_states [NUM_ITERATIONS] ;
    int actions [NUM_ITERATIONS] ;
    int num_iter = 0 ;
    srand ( time(NULL) );
    for (i=0 ;i< nrStates;i++)
    {
        belief[i]=  start[i];
    }
        real_state = choose_start_state () ;
        //cout << "real= " << real_state << " " ;
        //cout.flush();
        
        
    double b = clock();
    for (iter = 0 ; iter < NUM_ITERATIONS ; iter++)
    { 
        
        num_iter++ ;
        real_states[iter] = real_state ;
        max_utility = MIN ;
        action = 0 ;
        for (ac = 0 ; ac <  nrActions; ac++)
        {                          
            utility = 0 ;
            update_bel_action(ac);
            for (int s=0 ; s<nrStates; s++)
                utility = utility+ belief[s]*reward[states_actions[s][ac]][s][ac]+gamma*temp[s]*MDP_values_actions[s];
            
            if (utility>max_utility)
            {
                max_utility = utility ;
                action = ac ;
            }
         }
       // cout <<ac <<" ";
        //cout.flush();
         o = Solver::act_and_observe (real_state, action);       
         actions[iter] = action ;
         Solver::update_bel (action,o) ;
        // printf ("%d " , action ) ;
    }
    total_reward = 0 ; 
    float discount = 1 ;
    for (i=0 ; i < num_iter-1 ; i++)
    {
        discount = discount * gamma ;
        total_reward = total_reward + discount*reward[real_states[i+1]][real_states[i]][actions[i]] ;
    }
    double e = clock();
   //printf ("time= %f ",(e-b)/CLOCKS_PER_SEC);
   //printf ("last_state=%d  ", real_states[num_iter]);
    return total_reward ;
}
