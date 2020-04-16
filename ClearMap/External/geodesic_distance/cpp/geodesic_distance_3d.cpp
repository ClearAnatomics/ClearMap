#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include "util.h"
#include "geodesic_distance_3d.h"
using namespace std;

void insert_point_to_list(std::vector<Point3D> * list, int start_position,  Point3D p)
{
    int insert_idx = list->size();
    for(int i = start_position; i < list->size(); i++)
    {
        if(list->at(i).distance < p.distance)
        {
            insert_idx = i;
            break;
        }
    }
    list->insert(insert_idx + list->begin(), p);
}

void update_point_in_list(std::vector<Point3D> * list, Point3D p)
{
    int remove_idx = -1;
    for(int i = 0; i < list->size(); i++)
    {
        if(list->at(i).d == p.d && list->at(i).h == p.h && list->at(i).w == p.w)
        {
            remove_idx = i;
            break;
        }
    }
    list->erase(remove_idx + list->begin());
    insert_point_to_list(list, remove_idx, p);
}

void add_new_accepted_point(const float * img, float * distance, int * state,
                            std::vector<Point3D> * list,
                            int depth, int height, int width, Point3D p)
{
    int d = p.d;
    int h = p.h;
    int w = p.w;
    float p_value = get_pixel<float>(img, depth, height, width, d, h, w);
    float p_dis   = get_pixel<float>(distance, depth, height, width, d, h, w);
    for(int dd = -1; dd <= 1; dd++)
    {
        for(int dh = -1; dh <= 1; dh++)
        {
            for(int dw = -1; dw <= 1; dw++)
            {
                if(dd == 0 && dh == 0 && dw == 0) continue;
                int nd = dd + d;
                int nh = dh + h;
                int nw = dw + w;
                
                if(nd >=0 && nd < depth && nh >=0 && nh < height && nw >=0 && nw < width )
                {
                    int temp_state = get_pixel<int>(state, depth, height, width, nd, nh, nw);
                    if(temp_state == 0) continue;
                    float space_dis = sqrt(dd*dd + dh*dh + dw*dw);
                    float q_value   = get_pixel<float>(img, depth, height, width, nd, nh, nw);
                    float delta_dis = space_dis*abs(p_value - q_value);
                    float old_dis   = get_pixel<float>(distance, depth, height, width, nd, nh, nw);
                    
                    float new_dis   = p_dis + delta_dis;
                    if(new_dis < old_dis)
                    {
                        set_pixel<float>(distance, depth, height, width, nd, nh, nw, new_dis);
                        Point3D new_point;
                        new_point.distance = new_dis;
                        new_point.d = nd;
                        new_point.h = nh;
                        new_point.w = nw;
                        if(temp_state == 2){
                            set_pixel<int>(state, depth, height, width, nd, nh, nw, 1);
                            insert_point_to_list(list, 0, new_point);
                        }
                        else{
                            update_point_in_list(list, new_point);
                        }
                    }
                }
            } // end for dw
        } // end for dh
    }// end for dd
}

void geodesic3d_fast_marching(const float * img, const unsigned char * seeds, float * distance,
                         int depth, int height, int width)
{
    int * state = new int[depth * height * width];
    
    // point state: 0--acceptd, 1--temporary, 2--far away
    // get initial accepted set and far away set
    float init_dis;
    int init_state;
    for(int d = 0; d < depth; d++)
    {
        for(int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                unsigned char seed_type = get_pixel<unsigned char>(seeds, depth, height, width, d, h, w);
                if(seed_type > 0){
                    init_dis = 0.0;
                    init_state = 0;
                }
                else{
                    init_dis = 1.0e10;
                    init_state = 2;
                }
                set_pixel<float>(distance, depth, height, width, d, h, w, init_dis);
                set_pixel<int>(state, depth, height, width, d, h, w, init_state);
            }
        }
    }
    
    // get initial temporary set, and save them in a list
    std::vector<Point3D> temporary_list;
    temporary_list.reserve(depth * height * width);
    for(int d = 0; d < depth; d++)
    {
        for(int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                int temp_state = get_pixel<int>(state, depth, height, width, d, h, w);
                if(temp_state == 0)
                {
                    Point3D accepted_p;
                    accepted_p.distance = 0.0;
                    accepted_p.d = d;
                    accepted_p.h = h;
                    accepted_p.w = w;
                    add_new_accepted_point(img, distance, state, &temporary_list, depth, height, width, accepted_p);
                 }
            }
        }
    }
    // update temporary set until it is empty
    while(temporary_list.size() > 0){
        Point3D temp_point = temporary_list[temporary_list.size() - 1];
        temporary_list.pop_back();
        set_pixel<int>(state, depth, height, width, temp_point.d, temp_point.h, temp_point.w, 0);
        add_new_accepted_point(img, distance, state, &temporary_list, depth, height, width, temp_point);
    }
    delete [] state;
}

void geodesic3d_raster_scan(const float * img, const unsigned char * seeds, float * distance,
                              int depth, int height, int width, float lambda, int iteration)
{
    // point state: 0--acceptd, 1--temporary, 2--far away
    // get initial accepted set and far away set
    float init_dis;
    unsigned char seed_type;
    for(int d = 0; d < depth; d++)
    {
        for(int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                seed_type = get_pixel<unsigned char>(seeds, depth, height, width, d, h, w);
                init_dis = seed_type > 0 ? 0.0 : 1.0e10;
                set_pixel<float>(distance, depth, height, width, d, h, w, init_dis);
            }
        }
    }
    float sqrt3 = sqrt(3.0);
    float sqrt2 = sqrt(2.0);
    float sqrt1 = 1.0;
    for(int it = 0; it<iteration; it++)
    {
        // forward scann
        int dd_f[13] = {-1, -1, -1, -1, -1,  0,  0,  0,  0,  1,  1,  1,  1};
        int dh_f[13] = {-1, -1, -1,  0,  0, -1, -1, -1,  0, -1, -1, -1,  0};
        int dw_f[13] = {-1,  0,  1, -1,  0, -1,  0,  1, -1, -1,  0,  1, -1};
        int local_dis_f[13] = {sqrt3, sqrt2, sqrt3, sqrt2, sqrt1,
                               sqrt2, sqrt1, sqrt2, sqrt1,
                               sqrt3, sqrt2, sqrt3, sqrt2};
        // forward pass
        for(int d = 0; d < depth; d++)
        {
            for(int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    float p_dis = get_pixel<float>(distance, depth, height, width, d, h, w);
                    float p_value = get_pixel<float>(img, depth, height, width, d, h, w);
                    for(int i = 0; i < 13; i++)
                    {
                        int nd = d + dd_f[i];
                        int nh = h + dh_f[i];
                        int nw = w + dw_f[i];
                        if(nd < 0 || nd >= depth || nh < 0 || nh >= height || nw < 0 || nw >= width) continue;
                        float q_dis = get_pixel<float>(distance, depth, height, width, nd, nh, nw);
                        float q_value = get_pixel<float>(img, depth, height, width, nd, nh, nw);
                        float speed = (1.0 - lambda) + lambda/(abs(p_value - q_value) + 1e-5);
                        float delta_d = local_dis_f[i] / speed;
                        float temp_dis = q_dis + delta_d;
                        if(temp_dis < p_dis) p_dis = temp_dis;
                    }
                    set_pixel<float>(distance, depth, height, width, d, h, w, p_dis);
                }
            }
        }
        
        int dd_b[13] = {-1, -1, -1, -1,  0,  0,  0,  0,  1,  1,  1,  1,  1};
        int dh_b[13] = { 0,  1,  1,  1,  0,  1,  1,  1,  0,  0,  1,  1,  1};
        int dw_b[13] = { 1, -1,  0,  1,  1, -1,  0,  1,  0,  1, -1,  0,  1};
        int local_dis_b[13] = {sqrt2, sqrt3, sqrt2, sqrt3, sqrt1,
                               sqrt2, sqrt1, sqrt2, sqrt1,
                               sqrt2, sqrt3, sqrt2, sqrt3};
        // backward pass
        for(int d = depth - 1; d >= 0; d--)
        {
            for(int h = height - 1; h >= 0; h--)
            {
                for (int w = width - 1; w >= 0; w--)
                {
                    float p_dis = get_pixel<float>(distance, depth, height, width, d, h, w);
                    float p_value = get_pixel<float>(img, depth, height, width, d, h, w);
                    for(int i = 0; i < 13; i++)
                    {
                        int nd = d + dd_b[i];
                        int nh = h + dh_b[i];
                        int nw = w + dw_b[i];
                        if(nd < 0 || nd >= depth || nh < 0 || nh >= height || nw < 0 || nw >= width) continue;
                        float q_dis = get_pixel<float>(distance, depth, height, width, nd, nh, nw);
                        float q_value = get_pixel<float>(img, depth, height, width, nd, nh, nw);
                        float speed = (1.0 - lambda) + lambda/(abs(p_value - q_value) + 1e-5);
                        float delta_d = local_dis_b[i] / speed;
                        float temp_dis = q_dis + delta_d;
                        if(temp_dis < p_dis) p_dis = temp_dis;
                    }
                    set_pixel<float>(distance, depth, height, width, d, h, w, p_dis);
                }
            }
        }
    }
}
