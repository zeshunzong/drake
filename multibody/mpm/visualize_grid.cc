#include "drake/multibody/mpm/visualize_grid.h"
#include <fstream>
namespace drake {
namespace multibody {
namespace mpm {

void WriteGrid2obj(const std::string& filename, Grid& grid){

    std::ofstream myfile;
    myfile.open(filename);
    std::cout << myfile.is_open() << std::endl;
    myfile << "Writing this to a file.\n";
    myfile.close();


    std::cout<<filename<<std::endl;
    std::cout<<grid.get_h()<<std::endl;

    std::ofstream outfile ("test.txt");

    outfile << "aaaaaa!" << std::endl;
    outfile.close();
}

void WriteGridVelocity2obj(const std::string& filename, Grid& grid){
    std::cout<<filename<<std::endl;
    std::cout<<grid.get_h()<<std::endl;
}

}  // namespace mpm
}  // namespace multibody
}  // namespace drake