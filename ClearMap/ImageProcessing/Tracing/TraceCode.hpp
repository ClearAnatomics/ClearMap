// 3d tracing code
//__author__    = 'Christoph Kirst <ckirst@rockefeller.edu>'
//__license__   = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
//__copyright__ = 'Copyright (c) 2017 by Christoph Kirst, The Rockefeller University, New York City'

#include <iostream>
#include <stdio.h>

#include <vector>
#include <queue>
#include <map>
#include <algorithm>

#include <cmath>


using namespace std;

template<typename T, typename C>
class priority_queue_remove : public priority_queue<T, std::vector<T>, C>
{
  public:
    bool remove(const T& value) {
      typename priority_queue<T, std::vector<T> >::container_type::iterator it = std::find(this->c.begin(), this->c.end(), value);
      if (it != this->c.end()) {
        this->c.erase(it);
        std::make_heap(this->c.begin(), this->c.end(), this->comp);
        return true;
      } else {
        return false;
      }
    }

    void print() {
      std::cout << "Queue (" << this->size() << ")[";
      for (typename priority_queue<T, std::vector<T> >::container_type::iterator it = this->c.begin(); it != this->c.end(); it++) {
        std::cout << *(*it) << " ";
      }
      std::cout << "]";
    }

    void print_first() {
        std::cout << *(*(this->c.begin()));
    }

    void clear() {
      while(!this->empty()) {
        delete this->top();
        this->pop();
      }
    }
};


template<typename T>
class Point3D {
  public:
    T x, y, z;
  
  public:
    Point3D(T x_, T y_, T z_) {
      x = x_; y = y_; z = z_;
    }

    Point3D(const Point3D& pt) {
       x = pt.x; y = pt.y; z = pt.z;
    }

    Point3D(const vector<T>& v) {
       x = v[0]; y = v[1]; z = v[2];
    }

    Point3D() {
      x = y = z = -1;      
    }

    vector<T> toVector() const {
      vector<T> v();
      v.push(x); v.push(y); v.push(z);
      return v;
    }
  
    void fromVector(const vector<T>& v) {
      x = v[0]; y = v[1]; z = v[2];
    }
    
    bool operator==(const Point3D& right) const {
      return (x==right.x) && (y==right.y) && (z==right.z);
    }

    bool operator<(const Point3D& right) const {
      if (x < right.x) { return true; }
      else if (x > right.x) { return false; }

      if (y < right.y) { return true; }
      else if (y > right.y) { return false; }

      if (z < right.z) { return true; }
      else if (z > right.z) { return false; }

      return false;
    }
};

template<typename T>
std::ostream& operator<< (std::ostream & out, Point3D<T> const& point) {
    out << "(" << point.x << "," << point.y << "," << point.z << ")" ;
    return out ;
}


typedef Point3D<int> Point;
typedef Point3D<double> Scale;

const Scale one(1,1,1);

class Path {
  public:
    typedef vector<Point> points_t;
    points_t points;
    Scale scale;
    double f;

  public:
    Path(vector<Point> points_, Scale scale_ = one, double f_ = 0.0) {
      points = points_;
      scale  = scale_;
      f = f_;
    }

    Path(Scale scale_ = one, double f_ = 0.0) {
      points = vector<Point>();
      scale  = scale_;
      f = f_;
    }
    
    void addPoint(Point point) {
      points.push_back(point);
    }
    
    void reverse() {
      std::reverse(points.begin(), points.end());
    }

    void append(Path& path) {
      points.insert(points.end(), path.points.begin(), path.points.end());
      f += path.f;
    }

    void prepend(Path& path) {
      points.insert(points.begin(), path.points.begin(), path.points.end());
      f += path.f;
    }

    double length() {
      double l = 0.0;
      double dx,dy,dz;
      int s = points.size();
      for (int i = 0; i < s -1; i++) {
        dx = (points[i+1].x - points[i].x) * scale.x;
        dy = (points[i+1].y - points[i].y) * scale.y;
        dz = (points[i+2].z - points[i].z) * scale.z;
        l += sqrt(dx*dx + dy*dy + dz*dz);
      }
      return l;
    }

    int size() const {
      return points.size();
    }

    void clear() {
      points.clear();
      f = 0.0;     
    }
};

std::ostream& operator<< (std::ostream & out, Path const& path) {
  out << "Path(f=" << path.f << ")[" ;
  for (Path::points_t::const_iterator it = path.points.begin(); it != path.points.end(); it++){
    out << *it << ",";
  }
  out << "]" << std::endl;
  return out ;
}



enum search_status_t {
  FREE = 0, 
  OPEN_START,
  CLOSED_START,
  OPEN_GOAL,
  CLOSED_GOAL
};

class SearchNode {
  public:
    Point point;

    double g; // cost of the path so far (up to and including this node)
    double h; // heuristic esimate of the cost of going from here to the goal
    double f; // should always be the sum of g and h

    SearchNode* predecessor;

    search_status_t status;

  public:
    SearchNode(Point point_, double g_, double h_, SearchNode* predecessor_, search_status_t status_) { //}, int searchStatus_) {
      point = point_;
      g = g_;
      h = h_;
      f = g + h;
      predecessor = predecessor_;
      status = status_;
    }

    void from(SearchNode* node) {
      point = node->point;
      g = node->g;
      h = node->h;
      f = node->f;
      predecessor = node->predecessor;
      status = node->status;
    }
       
    void toPath(Path& path) {
      path.clear();
      SearchNode* s = this;
      //std::cout << "creating path"  << std::endl;  
      double ff = 0;
      int i = 0;
      while (s != NULL && i < 100) {
        path.addPoint(s->point);
        //std::cout << "point " << s->point << std::endl;  
        ff += s->f;
        s = s->predecessor;
        i++;
      }
      path.f = ff;
    }
  
    void toPathReversed(Path& path) {
      toPath(path);
      path.reverse();
    }

    bool operator<(const SearchNode& right) const {
      if (f < right.f) { return true; }
      else if (f > right.f) { return false; }

      return (point < right.point);
    }

    bool operator==(const SearchNode& right) const {
      return (point == right.point);
    }
};

std::ostream& operator<< (std::ostream & out, SearchNode const& node) {
  out << "SearchNode[" << node.point << ",g=" << node.g << ",h=" << node.h << ",f=" << node.f << ",s=" << node.status;
  if (node.predecessor != NULL) {
    std::cout << ",p=" << (node.predecessor)->point;
  } else {
    std::cout << ",p=NULL";
  }
  std::cout << "]";
  return out ;
}


struct SearchNodeComparator
{
    bool operator()(const SearchNode* lhs, const SearchNode* rhs) const
    {
      if (lhs->f < rhs->f) { return false; }
      else if (lhs->f > rhs->f) { return true; }

      if (lhs->point < rhs->point) { return false; }
      
      return true;
    }
};


template <typename source_t, typename index_t>
class Tracer {

  public:
    //image data
    source_t* source;
    index_t shape_x, shape_y, shape_z;
    index_t stride_x, stride_y, stride_z;

    //bool use_reward;
    source_t* reward;

    Scale scale;
    
    // result
    Path path;
    
    //parameter
    //bool reciprocal;
    double reciprocal_zero;
 
    double reward_multiplier;
    double minimal_reward;
   
    double cost_per_distance;
    double minimum_cost_per_distance;

    long max_step;

    bool verbose;

    // local data
    typedef priority_queue_remove<SearchNode*, SearchNodeComparator> search_node_queue_t;
    //typedef map<Point, SearchNode*> search_node_map_t;
    typedef map<index_t, SearchNode*> search_node_map_z_t;
    typedef map<index_t, search_node_map_z_t* > search_node_map_y_t;
    typedef map<index_t, search_node_map_y_t* > search_node_map_t;
  
  public:
    //constructors
//    Tracer(source_t* source_, index_t shape_x_, index_t shape_y_, index_t shape_z_,
//                                    index_t stride_x_,index_t stride_y_,index_t stride_z_,
//           source_t* reward_, Scale scale_ = one) {
//      setData(source_, shape_x_, shape_y_, shape_z_,
//                       stride_x_,stride_y_,stride_z_,
//              reward_);
//      
//      setDefault();
//      scale = scale_;
//      path = Path(scale, 0.0);
//    }

    Tracer() {
//      if (verbose) {
//        std::cout << "Constructing Tracer..." << std::endl;      
//      }

      setDefault();
    }
 
    void setDefault() {
      //reciprocal = true;
      reciprocal_zero = 0.5;
      
      cost_per_distance = 1.0;
      
      //if (use_reward) {
      minimum_cost_per_distance = 1 / 60.0;
      //} else {
      //minimum_cost = reciprocal ? (1 / 255.0) : 1;
      //minimum_cost_per_distance = 1 / 255.0;
      //}

      reward_multiplier = 4.0;

      minimal_reward = 0.2;
      
      verbose = true;

      scale = one;
      path = Path(scale, 0.0);

      max_step = -1;
    }
   
    void setData(source_t* source_, index_t shape_x_, index_t shape_y_, index_t shape_z_,
                                          index_t stride_x_,index_t stride_y_,index_t stride_z_,
                 source_t* reward_) {
      source = source_;
      shape_x = shape_x_;
      shape_y = shape_y_;
      shape_z = shape_z_;

      stride_x = stride_x_;
      stride_y = stride_y_;
      stride_z = stride_z_;

      //use_reward = use_reward_;
      reward = reward_;
    }
  
  public:
    // measures    
    virtual double costMovingTo(const Point& point, int dx, int dy, int dz) {
      //std::cout << "strides=(" << stride_x << "," << stride_y << "," << stride_z << ")" << std::endl;
      index_t index = point.x * stride_x + point.y * stride_y + point.z * stride_z;
      //std::cout << "index=" << index << std::endl;       
      double cost;
      //std::cout << "value=" << value_at_point << std::endl;
      
      //if (use_reward) {
      double measure = reward[index];
      if (measure <= minimal_reward) { measure = minimal_reward; }
      cost = 1 / (reward_multiplier * measure);
      //std::cout << "reward=" << measure << std::endl;
      //} else {
      //double value_at_point = source[index];
      //if (reciprocal) {
      //if (value_at_point != 0) {
      //    cost = 1.0 / value_at_point;
      //  } else {
      //    cost = 1.0 / reciprocal_zero;
      //  }
      //} else {
      //  cost = 256 - value_at_point;
      //}
      //}

      if (cost < minimum_cost_per_distance) {
        cost = minimum_cost_per_distance;
      }

      double dx2 = (dx * scale.x) * (dx * scale.x);
      double dy2 = (dy * scale.y) * (dy * scale.y);
      double dz2 = (dz * scale.z) * (dz * scale.z);

      return sqrt(dx2 + dy2 + dz2) * cost;
    }
    
    virtual double estimateCostToGoal(const Point& current, const Point& goal) {
      double dx = (goal.x - current.x) * scale.x;
      double dy = (goal.y - current.y) * scale.y;
      double dz = (goal.z - current.z) * scale.z;
      double d =  sqrt(dx * dx + dy * dy + dz * dz);
      return cost_per_distance * d;
    }

    virtual bool atGoal(const Point& point, const Point& goal) {
      return point == goal;
    }
  
  public:
    //search
    void addNodeToMap(SearchNode* n, search_node_map_t* sn_map) {
      index_t i = n->point.x;
      typename search_node_map_t::iterator it = sn_map->find(i);
      search_node_map_y_t* sn_map_y;
      if (it != sn_map->end()) {
        sn_map_y = it->second;
      } else {
        sn_map_y = new search_node_map_y_t();
        sn_map->operator[](i) = sn_map_y;
      }

      i = n->point.y;
      typename search_node_map_y_t::iterator ity = sn_map_y->find(i);
      search_node_map_z_t* sn_map_z;
      if (ity != sn_map_y->end()) {
        sn_map_z = ity->second;
      } else {
        sn_map_z = new search_node_map_z_t();
        sn_map_y->operator[](i) = sn_map_z;
      }

      sn_map_z->operator[](n->point.z) = n;
    }

    SearchNode* findNodeInMap(const Point& point, search_node_map_t* sn_map){
      typename search_node_map_t::iterator it = sn_map->find(point.x);
      if (it == sn_map->end()) {
        return NULL;
      }

      typename search_node_map_y_t::iterator ity = it->second->find(point.y);
      if (ity == it->second->end()) {
        return NULL;
      }
      
      typename search_node_map_z_t::iterator itz = ity->second->find(point.z);
      if (itz == ity->second->end()) {
        return NULL;
      } else {
        return itz->second;
      }
    }
    
    void clearMap(search_node_map_t* sn_map) { // this routine has a name that reminds me of another great program:)
      for (typename search_node_map_t::iterator it = sn_map->begin(); it != sn_map->end(); it++) {
        for (typename search_node_map_y_t::iterator ity = it->second->begin(); ity != it->second->end(); ity++) {
          delete ity->second;
        }
        delete it->second;
      }
    }

    void addNode(SearchNode* n, search_node_queue_t* sn_queue, search_node_map_t* sn_map) {
      sn_queue->push(n);
      //sn_map->operator[](n->point) = n;
	 addNodeToMap(n, sn_map);
    }

    //main tracing routine
    int search(Point& start, Point& stop, bool bidirectional) {

//      if (verbose) {
//        std::cout << "Searching path..." << std::endl;      
//      }
          
      // init
      search_node_queue_t closed_start; 
      search_node_queue_t open_start;
  
      search_node_queue_t closed_goal;
      search_node_queue_t open_goal;
  
      search_node_map_t map_start;
      search_node_map_t map_goal;

      search_node_queue_t* closed_current;
      search_node_queue_t* open_current;
      
      search_node_map_t* map_current;
      search_node_map_t* map_other;

      bool from_start = true;
  
      Point goal_current;

      search_status_t OPEN_CURRENT, CLOSED_CURRENT;

      path.clear();
      
      SearchNode* snstart = new SearchNode(start, 0, estimateCostToGoal(start, stop ), NULL, OPEN_START);
      SearchNode* sngoal  = new SearchNode(stop , 0, estimateCostToGoal(stop , start), NULL, OPEN_GOAL );
      
      addNode(snstart, &open_start, &map_start);
      addNode(sngoal,  &open_goal,  &map_goal);

      int step = 0;

      while (!(open_start.empty()) || (bidirectional && (!(open_goal.empty())))) {

        if (max_step > 0) {
          step++;
          if (step > max_step) { return -1; }
        }

        if (bidirectional) {
          from_start = open_start.size() <= open_goal.size();
        }
        
        
        if (verbose) {
          //std::cout << "---------------------------------------------------------" << std::endl;   
          std::cout << "Next iteration " << step << " from " << (from_start ? "start" : "goal") << std::endl;  
          std::cout << "Queues: open start:" << open_start.size() << " open goal:" << open_goal.size() << std::endl;
          std::cout << "open start top: ";
          open_start.print_first();
          std::cout << "open goal top: "; 
          open_goal.print_first();
          std::cout << std::endl;
        }
        
        open_current   = from_start ? &open_start   : &open_goal;
        closed_current = from_start ? &closed_start : &closed_goal;

        map_current = from_start ? &map_start : &map_goal;
        map_other   = from_start ? &map_goal  : &map_start;

        goal_current = from_start ? stop  : start;

        OPEN_CURRENT = from_start ? OPEN_START : OPEN_GOAL;
        CLOSED_CURRENT = from_start ? CLOSED_START : CLOSED_GOAL;

//        if (verbose) {
//          std::cout << "Queues: open:" << open_current->size() << " closed:" << closed_current->size() << std::endl;
//          std::cout << "open curent:" << std::endl;
//          open_current->print();
//          std::cout << std::endl << "closed current:" << std::endl;
//          closed_current->print();
//          std::cout << std::endl;
//        }
        
        if (open_current->size() == 0)
		continue;

        SearchNode* node = open_current->top();
        open_current->pop();

        if (verbose) {
          std::cout << *node << std::endl;    
        }

        // found the goal?
        if (this->atGoal(node->point, goal_current)) {
//          if (verbose) {
//             std::cout << "Goal found!" << std::endl;
//          }
		if (from_start) {
            node->toPath(path);
          } else {
            node->toPathReversed(path);
          }

          //cleanup
          closed_start.clear();
          open_start.clear();
          closed_goal.clear();
          open_goal.clear();
          clearMap(&map_start);
          clearMap(&map_goal); 
          return 1;
        }

        node->status = CLOSED_CURRENT;
        addNode(node, closed_current, map_current);

//        if (verbose) {
//          std::cout << "Node added to closed queue" << std::endl; 
//          std::cout << "closed current:" << std::endl;
//          closed_current->print();
//          std::cout << std::endl;
//        }

        // search neighbours 
        for (int dz = -1; dz <= 1; dz++) {
          int z_new = node->point.z + dz;
          if (z_new < 0 || z_new >= shape_z)
            continue;

          for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {

              if ((dx == 0) && (dx == 0) && (dz == 0))
                continue;

              int x_new = node->point.x + dx;
              int y_new = node->point.y + dy;

              if (x_new < 0 || x_new >= shape_x)
                continue;

              if (y_new < 0 || y_new >= shape_y)
                continue;

              Point point_new(x_new, y_new, z_new);

              double h_new = this->estimateCostToGoal(point_new, goal_current);
              double g_new = node->g + this->costMovingTo(point_new, dx, dy, dz);  
              //double f_new = h_new + g_new;  

              SearchNode* node_new = new SearchNode(point_new, g_new, h_new, node, FREE);

//              if (verbose) {
//                std::cout << "New node: " << *node_new << std::endl;    
//                std::cout.flush();
//              }
              
              //search_node_map_t::iterator it = map_current->find(point_new);
              SearchNode* searched = findNodeInMap(point_new, map_current);

              //if (it == map_current->end()) { // not in list
              if (searched == NULL) { // not in list
//                if (verbose) {
//                  std::cout << "Node not in list." << std::endl;    
//                }
                node_new->status = OPEN_CURRENT;
                addNode(node_new, open_current, map_current);
//                if (verbose) {
//                  std::cout << "Noded added to open current." << std::endl << "open current:" << std::endl;
//                  open_current->print();
//                  std::cout << std::endl;
//                }

              } else {  //already path to new position but this one might be better 
//                if (verbose) {
//                  std::cout << "Node in list, checking for a better path." << std::endl;    
//                }

                //SearchNode* searched = it->second;                     
                if (node_new->f < searched->f) {
//                  if (verbose) {
//                    std::cout << "Better path found: " << *searched << std::endl;    
//                  }

                  if (searched->status == OPEN_CURRENT) {
                    open_current->remove(searched);
                  } else if (searched->status == CLOSED_CURRENT) {
                    closed_current->remove(searched);
                  }
                  node_new->status = OPEN_CURRENT;
                  searched->from(node_new); // effectively adds new node to map
                  open_current->push(searched);
                  delete node_new;

//                  if (verbose) {
//                    std::cout << "Updated open: " << std::endl;
//                    open_current->print();   
//                    std::cout << std::endl << "Updated closed: " << std::endl;
//                    closed_current->print();                               
//                  }
                } else {
                  delete node_new;
                }
              }

              if (bidirectional) {

                //it = map_other->find(point_new);
                SearchNode* searched = findNodeInMap(point_new, map_other);
                //if (it != map_other->end()) {
                if (searched != NULL) {
//                  if (verbose) {
//                    std::cout << "Searching for path!" << std::endl;    
//                  }
                  //SearchNode* searched = it->second;    
                  if (searched->status == CLOSED_START || searched->status == CLOSED_GOAL) {
//                    if (verbose) {
//                      std::cout << "Found path!" << std::endl;   
//                      std::cout << "current open queue" << std::endl;
//                      open_current->print();
//                      std::cout << std::endl << "current closed queue" << std::endl;
//                      closed_current->print(); 
//                      std::cout << std::endl;
//                    }
                    if (from_start) {
                      node->toPath(path);
                      //std::cout << "start " << path << std::endl;    
                      Path path_from_goal;
                      searched->toPathReversed(path_from_goal);
                      //std::cout << "start2  " << path_from_goal << std::endl;  
                      path.append(path_from_goal);
                    } else {
                      searched->toPath(path);
                      //std::cout << "goal " << path << std::endl;  
                      Path path_from_goal;
                      node->toPathReversed(path_from_goal);
                      //std::cout << "goal2  " << path_from_goal << std::endl;  
                      path.append(path_from_goal);
                    }	
//                    if (verbose) {
//                      std::cout << path << std::endl;    
//                    }
            		
                    closed_start.clear();
                    open_start.clear();
                    closed_goal.clear();
                    open_goal.clear();
                    clearMap(&map_start);
                    clearMap(&map_goal);           
                    return 1;
                  }
                }
              }
            } //for
          } 
        }
      } //while
      //printf('this shoud never happen!');

      closed_start.clear();
      open_start.clear();
      closed_goal.clear();
      open_goal.clear();
      clearMap(&map_start);
      clearMap(&map_goal);      
      return -1;
    } //search

  // intreface to python/cython
  int run(source_t* source_, index_t shape_x_, index_t shape_y_, index_t shape_z_,
                             index_t stride_x_,index_t stride_y_,index_t stride_z_,
          source_t* reward_,
          index_t start_x_, index_t start_y_, index_t start_z_,
          index_t goal_x_,  index_t goal_y_,  index_t goal_z_) {

    //std::cout << "init shape=" << shape_x_ << "," << shape_y_ << "," << shape_z_;
    //std::cout << ", strides=" << stride_x_ << "," << stride_y_ << "," << stride_y_ << std::endl;
    setData(source_, shape_x_, shape_y_, shape_z_,
                     stride_x_,stride_y_,stride_z_,
            reward_);

    Point start(start_x_, start_y_, start_z_);
    Point goal(goal_x_, goal_y_, goal_z_);

    return search(start, goal, true);  
  }
  
  int getPathSize() {
    return path.size();
  }

  double getPathQuality() {
    return path.f;
  }

  void getPath(index_t* path_array) {
    index_t stride_dim = 3;
    Point pp;
    index_t i = 0;      
    for (Path::points_t::iterator it = path.points.begin(); it != path.points.end(); it++) {
      path_array[i    ] = it->x;
      path_array[i + 1] = it->y;
      path_array[i + 2] = it->z;
      i+=stride_dim;
    }  
  } 
}; // Tracer



template <typename source_t, typename index_t, typename mask_t>
class TracerToMask : public Tracer<source_t, index_t> {

  public:
    //mask
    mask_t* mask;

    TracerToMask() {
      this->setDefault();
    }
    
    // intreface to python/cython
    int run(source_t* source_, index_t shape_x_, index_t shape_y_, index_t shape_z_,
                               index_t stride_x_,index_t stride_y_,index_t stride_z_,
            source_t* reward_,
            index_t start_x_, index_t start_y_, index_t start_z_,
            mask_t* mask_) {
  
      //std::cout << "init shape=" << shape_x_ << "," << shape_y_ << "," << shape_z_;
      //std::cout << ", strides=" << stride_x_ << "," << stride_y_ << "," << stride_y_ << std::endl;
      this->setData(source_, shape_x_, shape_y_, shape_z_,
                             stride_x_,stride_y_,stride_z_,
                    reward_);
  
      mask = mask_;
  
      Point start(start_x_, start_y_, start_z_);
      Point goal(0, 0, 0);
  
      return this->search(start, goal, false);  
    }


    double estimateCostToGoal(const Point& current, const Point& goal) {
      index_t index = current.x * this->stride_x + current.y * this->stride_y + current.z * this->stride_z;
      double d = mask[index];
      //std::cout << "cost to goal at " << current << " = " << mask[index] << std::endl; 
      return this->cost_per_distance * d;
    }

    bool atGoal(const Point& point, const Point& goal) {
      index_t index = point.x * this->stride_x + point.y * this->stride_y + point.z * this->stride_z;
      //std::cout << "at goal " << point << " = " << mask[index] << std::endl; 
      return mask[index] == 0;
    }
}; // TraceToMask
