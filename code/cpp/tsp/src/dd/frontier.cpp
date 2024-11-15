#include "frontier.hpp"

//
// Auxiliary comparator
//
struct SolComp
{
    bool operator()(const ObjType *solA, const ObjType *solB)
    {
        for (int i = 0; i < NOBJS; ++i)
        {
            if (solA[i] != solB[i])
            {
                return (solA[i] > solB[i]);
            }
        }
        return (solA[0] > solB[0]);
    }
};


//
// Print elements in set
//
void ParetoFrontier::print_frontier()
{
    for(SolutionList::iterator it=sols.begin(); it != sols.end(); ++it){
        (*it).print_x();
        (*it).print_obj();
    }    
}


//
// Add element to set
//
void ParetoFrontier::add(Solution& new_sol)
{
    bool must_add = true;
    bool dominates;
    bool dominated;
    // for (int i = 0; i < sols.size(); i += NOBJS)
    for (SolutionList::iterator it = sols.begin(); it != sols.end();)
    {
        // check status of foreign solution w.r.t. current frontier solution
        dominates = true;
        dominated = true;
        for (int o = 0; o < NOBJS && (dominates || dominated); ++o)
        {
            dominates &= (new_sol.obj[o] >= (*it).obj[o]);
            dominated &= (new_sol.obj[o] <= (*it).obj[o]);
        }
        if (dominated)
        {
            // if foreign solution is dominated, nothing needs to be done
            return;
        }
        else if (dominates)
        {
            // solution dominates iterate
            it = sols.erase(it);
        }
        else
        {
            ++it;
        }
    }
    sols.insert(sols.end(), new_sol);

}


//
// Merge pareto frontier into existing set considering shift
//
void ParetoFrontier::merge(ParetoFrontier &frontier, const ObjType *shift, int last_city)
{
    // last position to check
    // int end = sols.size();
    // if current solution set was modified
    // bool modified = false;
    // add each solution from frontier set
    bool must_add;
    bool dominates;
    bool dominated;
    Solution dummy;
    SolutionList::iterator end = sols.insert(sols.end(), dummy);    
    // for (int j = 0; j < frontier.sols.size(); j += NOBJS)
    // {
    for (SolutionList::iterator itParent = frontier.sols.begin();
         itParent != frontier.sols.end();
         ++itParent)
    {
    
        // update auxiliary
        for (int o = 0; o < NOBJS; ++o)
        {
            aux[o] = (*itParent).obj[o] + shift[o];
        }
        must_add = true; // if solution must be added to set
        // for (int i = 0; i < end; i += NOBJS)
        // {
        for (SolutionList::iterator itCurr = sols.begin();
             itCurr != end;)
        {
            // check if solution has been removed
            // if (sols[i] == DOMINATED)
            // {
            //     continue;
            // }
            // check status of foreign solution w.r.t. current frontier solution
            dominates = true;
            dominated = true;
            for (int o = 0; o < NOBJS && (dominates || dominated); ++o)
            {
                dominates &= (aux[o] >= (*itCurr).obj[o]);
                dominated &= (aux[o] <= (*itCurr).obj[o]);
            }
            if (dominated)
            {
                // if foreign solution is dominated, just stop loop
                must_add = false;
                break;
            }
            else if (dominates)
            {
                // // // if foreign solution dominates, replace current solution
                // // must_add = false;
                // // std::copy(aux, aux+NOBJS, &sols[i]);
                // // // search for domination in the remaining part of the array
                // // for (int k = i+NOBJS; k < end; k += NOBJS) {
                // //     if (sols[k] != DOMINATED) {
                // //         if (AdominatesB<NOBJS>(aux, &sols[k])) {
                // //            sols[k] = DOMINATED;
                // //         }
                // //     }
                // // }
                // // break;

                // // if foreign solution dominates, check if replacement is necessary
                // if (must_add)
                // {
                //     // solution has not been added - just replace current iterate
                //     std::copy(aux, aux + NOBJS, &sols[i]);
                //     must_add = false;
                // }
                // else
                // {
                //     // if already added, mark array as "to erase"
                //     sols[i] = DOMINATED;
                //     // modified = true;
                // }
                // must_add = false;
                itCurr = sols.erase(itCurr);
            }
            else{
                ++itCurr;
            }
        }
        // if solution has not been added already, append element to the end
        if (must_add)
        {
            // Create new solution object
            Solution new_sol((*itParent).x, (*itParent).obj);
            new_sol.x.push_back(last_city);
            for (int o = 0; o < NOBJS; ++o)
            {
                new_sol.obj[o] = aux[o];
            }
            // Push new solution at the end of the current solution list
            sols.push_back(new_sol);

            // sols.insert(sols.end(), aux, aux + NOBJS);
        }
    }
    // if (modified) {
    // remove_empty();
    //}
    sols.erase(end);
}


//
// Merge pareto frontier into existing set considering shift
//
void ParetoFrontier::merge(ParetoFrontier &frontier, Solution& offset_sol, bool offset_from_bu)
{
    // last position to check
    // int end = sols.size();
    // if current solution set was modified
    // bool modified = false;
    // add each solution from frontier set
    bool must_add;
    bool dominates;
    bool dominated;
    Solution dummy;
    SolutionList::iterator end = sols.insert(sols.end(), dummy);    
    // for (int j = 0; j < frontier.sols.size(); j += NOBJS)
    // {

    // X variable order if the offset is from the bottom-up set
    if (offset_from_bu){
        reverse(offset_sol.x.begin(), offset_sol.x.end());
    }

    for (SolutionList::iterator itParent = frontier.sols.begin();
         itParent != frontier.sols.end();
         ++itParent)
    {
    
        // update auxiliary
        for (int o = 0; o < NOBJS; ++o)
        {
            aux[o] = (*itParent).obj[o] + offset_sol.obj[o];
        }
        must_add = true; // if solution must be added to set
        // for (int i = 0; i < end; i += NOBJS)
        // {
        for (SolutionList::iterator itCurr = sols.begin();
             itCurr != end;)
        {
            // check if solution has been removed
            // if (sols[i] == DOMINATED)
            // {
            //     continue;
            // }
            // check status of foreign solution w.r.t. current frontier solution
            dominates = true;
            dominated = true;
            for (int o = 0; o < NOBJS && (dominates || dominated); ++o)
            {
                dominates &= (aux[o] >= (*itCurr).obj[o]);
                dominated &= (aux[o] <= (*itCurr).obj[o]);
            }
            if (dominated)
            {
                // if foreign solution is dominated, just stop loop
                must_add = false;
                break;
            }
            else if (dominates)
            {
                // // // if foreign solution dominates, replace current solution
                // // must_add = false;
                // // std::copy(aux, aux+NOBJS, &sols[i]);
                // // // search for domination in the remaining part of the array
                // // for (int k = i+NOBJS; k < end; k += NOBJS) {
                // //     if (sols[k] != DOMINATED) {
                // //         if (AdominatesB<NOBJS>(aux, &sols[k])) {
                // //            sols[k] = DOMINATED;
                // //         }
                // //     }
                // // }
                // // break;

                // // if foreign solution dominates, check if replacement is necessary
                // if (must_add)
                // {
                //     // solution has not been added - just replace current iterate
                //     std::copy(aux, aux + NOBJS, &sols[i]);
                //     must_add = false;
                // }
                // else
                // {
                //     // if already added, mark array as "to erase"
                //     sols[i] = DOMINATED;
                //     // modified = true;
                // }
                // must_add = false;
                itCurr = sols.erase(itCurr);
            }
            else{
                ++itCurr;
            }
        }
        // if solution has not been added already, append element to the end
        if (must_add)
        {
            // Create new solution object
            if (offset_from_bu){
                Solution new_sol((*itParent).x, (*itParent).obj);
                new_sol.x.insert(new_sol.x.end(), offset_sol.x.begin(), offset_sol.x.end());
                for(int i =0; i<NOBJS; ++i){
                    new_sol.obj[i] = aux[i];
                }
                sols.push_back(new_sol);

            }else{
                Solution new_sol(offset_sol.x, offset_sol.obj);   
                reverse((*itParent).x.begin(), (*itParent).x.end());
                new_sol.x.insert(new_sol.x.end(), (*itParent).x.begin(), (*itParent).x.end());
                for(int i =0; i<NOBJS; ++i){
                    new_sol.obj[i] = aux[i];
                }
                sols.push_back(new_sol);
             
            }            
        }
    }
    // if (modified) {
    // remove_empty();
    //}
    sols.erase(end);
}






//
// Convolute two nodes from this set to this one
//
void ParetoFrontier::convolute(ParetoFrontier &fA, ParetoFrontier &fB)
{
    if (fA.sols.size() < fB.sols.size())
    {
        for (SolutionList::iterator solA = fA.sols.begin(); solA != fA.sols.end(); ++solA)
        {
            // copy(fA.sols.begin() + j, fA.sols.begin() + j + NOBJS, auxB);
            merge(fB, (*solA), false);
        }
    }
    else
    {
        for (SolutionList::iterator solB = fB.sols.begin(); solB != fB.sols.end(); ++solB)
        {
            // copy(fB.sols.begin() + j, fB.sols.begin() + j + NOBJS, auxB);
            merge(fA, (*solB), true);
        }
    }
}


//
// Obtain sum of points
//
ObjType ParetoFrontier::get_sum()
{
    ObjType sum = 0;
    // for (int i = 0; i < sols.size(); ++i)
    // {
    for(SolutionList::iterator it=sols.begin(); it!=sols.end(); ++it){
        for(int o=0; o <NOBJS; ++o){
            sum += (*it).obj[o];
        }
    }
    return sum;
}



map<string, vector<vector<int>>> ParetoFrontier::get_frontier()
{
    // cout << sols.size() << endl;
    vector<vector<int>> x_sols;
    vector<vector<int>> z_sols;
    x_sols.reserve(sols.size());
    z_sols.reserve(sols.size());
    for (SolutionList::iterator it = sols.begin(); it != sols.end(); ++it)
    {
        x_sols.push_back((*it).x);
        z_sols.push_back((*it).obj);
    }

    map<string, vector<vector<int>>> frontier;
    frontier.insert({"x", x_sols});
    frontier.insert({"z", z_sols});

    return frontier;
}



// //
// // Sort array in decreasing order
// //
// inline void ParetoFrontier::sort_decreasing()
// {
//     const int num_sols = get_num_sols();
//     while (elems.size() < num_sols)
//     {
//         elems.push_back(new ObjType[NOBJS]);
//     }
//     int ct = 0;
//     for (int i = 0; i < sols.size(); i += NOBJS)
//     {
//         copy(sols.begin() + i, sols.begin() + i + NOBJS, elems[ct++]);
//     }
//     sort(elems.begin(), elems.begin() + num_sols, SolComp());
//     ct = 0;
//     for (int i = 0; i < num_sols; ++i)
//     {
//         copy(elems[i], elems[i] + NOBJS, sols.begin() + ct);
//         ct += NOBJS;
//     }
// }

// //
// // Check consistency
// //
// inline bool ParetoFrontier::check_consistency()
// {
//     for (int i = 0; i < sols.size(); i += NOBJS)
//     {
//         assert(sols[i] != DOMINATED);
//         for (int j = i + NOBJS; j < sols.size(); j += NOBJS)
//         {
//             // check status of foreign solution w.r.t. current frontier solution
//             bool dominates = true;
//             bool dominated = true;
//             for (int o = 0; o < NOBJS && (dominates || dominated); ++o)
//             {
//                 dominates &= (sols[i + o] >= sols[j + o]);
//                 dominated &= (sols[i + o] <= sols[j + o]);
//             }
//             assert(!dominates);
//             assert(!dominated);
//             if (dominates || dominated)
//             {
//                 return false;
//             }
//         }
//     }
//     return true;
// }

// //
// // Check if solution is dominated by any element of this set
// //
// inline bool ParetoFrontier::is_sol_dominated(const ObjType *sol, const ObjType *shift)
// {
//     bool dominated = false;
//     for (int i = 0; i < sols.size() && !dominated; i += NOBJS)
//     {
//         dominated = AdominatedB<NOBJS>(sol, &sols[i]);
//     }
//     return dominated;
// }