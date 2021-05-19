#define _CRT_SECURE_NO_WARNINGS


// C/C++ includes
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <type_traits>

// dlib includes
#include "dlib/rand.h"
#include "dlib/matrix.h"
#include "dlib/pixel.h"
#include "dlib/image_io.h"
#include "dlib/image_transforms.h"
#include "dlib/opencv.h"
#include "dlib/gui_widgets.h"
#include <dlib/optimization.h>
#include <dlib/global_optimization.h>
#include <dlib/numeric_constants.h>
#include <dlib/geometry.h>

// OpenCV includes
#include <opencv2/core.hpp>           
#include <opencv2/highgui.hpp>     
#include <opencv2/imgproc.hpp> 
#include <opencv2/video.hpp>
#include <opencv2/imgcodecs.hpp>

// custom includes
#include "get_current_time.h"
#include "get_platform.h"
#include "num2string.h"
#include "file_parser.h"
#include "file_ops.h"
//#include "make_dir.h"
#include "dlib_matrix_threshold.h"
#include "gorgon_capture.h"
#include "modulo.h"

#include "pso.h"
#include "bot_particle.h"
#include "connected_components.h"

//#define M_PI 3.14159265358979323846

// -------------------------------GLOBALS--------------------------------------

extern const uint32_t array_depth=3;

dlib::rand rnd(time(NULL));

// ----------------------------------------------------------------------------------------
/*
using car_net_type = dlib::loss_mean_squared_multioutput<dlib::htan<dlib::fc<2,
    dlib::multiply<dlib::fc<5,
    dlib::multiply<dlib::fc<20,
    dlib::multiply<dlib::fc<80,
    dlib::input<dlib::matrix<float>>
    >> >> >> >>>;
car_net_type c_net(dlib::multiply_(0.0001), dlib::multiply_(0.5), dlib::multiply_(0.5));
*/

using car_net_type = dlib::loss_mean_squared_multioutput<dlib::htan<dlib::fc<2,
    dlib::multiply<dlib::fc<20,
    dlib::multiply<dlib::fc<120,
    dlib::input<dlib::matrix<float>>
    >> >> >>>;
car_net_type c_net(dlib::multiply_(0.0001), dlib::multiply_(0.5));

dlib::image_window win;
dlib::matrix<dlib::rgb_pixel> color_map;

extern const uint64_t fc4_size;
extern const uint64_t fc3_size;
extern const uint64_t fc2_size;
extern const uint64_t fc1_size;

// ----------------------------------------------------------------------------------------

class vehicle
{

public:
	
	uint8_t threshold = 240;
	double heading;
    const unsigned long width = 6;
    const unsigned long length = 8;
    const uint32_t radius = 1;

    const double L = 6;     // wheel base
    const double r = 1;      // wheel radius

    double points;

    dlib::point C;

	float max_range = 80.0;
    //std::vector<double> detection_angles = { -135, -90, -45, 0, 45, 90, 135 };
    //std::vector<double> detection_angles = {-135, -125, -115, -105, -95, -85, -75, -65, -55, -45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135};
    //std::vector<double> detection_angles = { -90.0, -45.0, -22.5, -11.0, -0.5, 0.5, 11.0, 22.5, 45.0, 90.0 };
    std::vector<double> detection_angles = { -60.0, -45.0, -22.5, -11.0, -0.5, 0.5, 11.0, 22.5, 45.0, 60.0 };

    //std::vector<double> detection_angles;

	std::vector<float> detection_ranges;

    vehicle(dlib::point C_, double heading_)
    {
        C = C_;

        heading = heading_*(dlib::pi / 180.0);

        points = 0.0;

        //for (double idx = -90; idx <= 90; idx += 22.5)
        for (uint32_t idx = 0; idx < detection_angles.size(); ++idx)
        {
            detection_angles[idx] = detection_angles[idx] * (dlib::pi / 180.0);
            //detection_angles.push_back(idx * (dlib::pi / 180.0));
        }

        detection_ranges.resize(detection_angles.size());
    }

	void get_ranges(dlib::matrix<uint8_t> map, dlib::matrix<uint8_t> &map2)
	{
		uint32_t idx;
		uint32_t x, y;

        uint32_t map_width = map.nc();
        uint32_t map_height = map.nr();

        //double h = heading + dlib::pi;

        dlib::assign_image(map2, map);

        map2(C.y(), C.x()) = 200;
		
		for(idx=0; idx< detection_angles.size(); ++idx)
		{
			detection_ranges[idx] = 1.0;
			
			for(uint32_t r=1; r<max_range; ++r)
			{

                x = (uint32_t)floor(r*std::cos(heading + detection_angles[idx]) + C.x() + 0.5);
                y = (uint32_t)floor(r*std::sin(heading + detection_angles[idx]) + C.y() + 0.5);

				
                if (x<0 || x>(map_width-1))
                {
                    detection_ranges[idx] = r/ max_range;
                    break;
                }

                if (y<0 || y>(map_height-1))
                {
                    detection_ranges[idx] = r/ max_range;
                    break;
                }

				if(map(y,x) > threshold)
				{
					detection_ranges[idx] = r/ max_range;
                    break;
				}
                map2(y, x) = 128;

			}
			
		}
	}   // end of get_ranges
	
    // ----------------------------------------------------------------------------------------

    //void move(double vl, double vr)
    //{
    //    double w = (vr - vl) / L;
    //    double h = heading + dlib::pi;
    //    double R = 0.0;
    //    
    //    long x_p = 0;
    //    long y_p = 0;

    //    if (std::abs(vr - vl) < 0.01)
    //    {
    //        x_p = C.x() + r * vr * std::cos(h);
    //        y_p = C.y() + r * vr * std::sin(h);
    //    }
    //    else
    //    {
    //        R = (L / 2.0) * (vr + vl);

    //        double ICCx = C.x() - R * std::sin(h);
    //        double ICCy = C.y() + R * std::cos(h);

    //        x_p = R * std::sin(h) * std::cos(w) + R * std::cos(h) * std::sin(w) + ICCx;
    //        y_p = R * std::sin(h) * std::sin(w) - R * std::cos(h) * std::cos(w) + ICCy;
    //    }

    //    C = dlib::point(x_p, y_p);

    //    heading += w;

    //}   // end of move


    void move(double fb, double lr)
    {

        //heading += lr * (dlib::pi * 0.0027777777778);    // 1/360
        heading += lr * (dlib::pi * 0.0055555555556);    // 2/360 
        //heading += lr * (dlib::pi * 0.0083333333333);    // 3/360 
        //heading += lr * (dlib::pi * 0.0138888888889);    // 5/360

        if (heading >= 2.0*dlib::pi)
            heading -= 2.0*dlib::pi;
        else if(heading <= -2.0*dlib::pi)
            heading += 2.0*dlib::pi;

        long x_p = C.x() + std::floor(fb * std::cos(heading) + 0.5);
        long y_p = C.y() + std::floor(fb * std::sin(heading) + 0.5);

        C = dlib::point(x_p, y_p);

    }   // end of move

    // ----------------------------------------------------------------------------------------
    
    bool test_for_crash(dlib::matrix<uint8_t> map)
    {
        bool crash = false;

        //double h = heading - dlib::pi / 2.0;
        double h = heading;

        long Lx = C.x() - floor((L / 2.0) * std::cos(h) + 0.5);
        long Ly = C.y() - floor((L / 2.0) * std::sin(h) + 0.5);

        long Rx = C.x() + floor((L / 2.0) * std::cos(h) + 0.5);
        long Ry = C.y() + floor((L / 2.0) * std::sin(h) + 0.5);

        //map(C.y(), C.x()) = 200;

        if ((C.x() < 1) || C.x() >= map.nc() - 1)
            crash = true;

        else if ((C.y() < 1) || C.y() >= map.nr() - 1)
            crash = true;

        else if ((Lx < 1) || Lx >= map.nc()-1)
            crash = true;

        else if ((Ly < 1) || Ly >= map.nr()-1)
            crash = true;

        else if ((Rx < 1) || Rx >= map.nc()-1)
            crash = true;

        else if ((Ry < 1) || Ry >= map.nr()-1)
            crash = true;

        else if (map(Ly, Lx) > threshold)
        {
            crash = true;
            map(Ly, Lx) = 100;
            map(Ry, Rx) = 150;
        }
        else if (map(Ry, Rx) > threshold)
        {
            crash = true;
            map(Ly, Lx) = 100;
            map(Ry, Rx) = 150;
        }
        else
        {
            map(Ly, Lx) = 100;
            map(Ry, Rx) = 150;
        }

        return crash;

    }   // end of test_for_crash

    void check_for_points(dlib::matrix<uint8_t> &map)
    {
        if (map(C.y(), C.x()) == 85)
        {
            points += 10;
            clear_line(map);
        }
        else if (map(C.y(), C.x()) == 170)
        {
            points += 100;
            clear_line(map);
        }
    }


private:

    void clear_line(dlib::matrix<uint8_t> &map)
    {

        uint8_t T = 0;
        uint8_t L = 0;
        connected_set(C, map, L, map(C.y(), C.x()));

        //long offset = 9;

        //long x1 = std::min(std::max(C.x() - offset, 0L), map.nc() - 1); 
        //long y1 = std::min(std::max(C.y() - offset, 0L), map.nr() - 1); 

        //long x2 = std::min(C.x() + offset, map.nc() - 1); 
        //long y2 = std::min(C.y() + offset, map.nr() - 1); 

        //long x1 = std::min(std::max(C.x() + (long)(offset * std::cos(heading + (dlib::pi / 2.0))), 0L), map.nc() - 1); 
        //long y1 = std::min(std::max(C.y() + (long)(offset * std::sin(heading + (dlib::pi / 2.0))), 0L), map.nr() - 1); 

        //long x2 = std::min(C.x() + (long)(offset * std::cos(heading - (dlib::pi / 2.0))), map.nc() - 1); 
        //long y2 = std::min(C.y() + (long)(offset * std::sin(heading - (dlib::pi / 2.0))), map.nr() - 1); 

        //dlib::point p1(x1,y1), p2(x2,y2);

        //if (x1 > x2)
        //{
        //    p1.x() = x2;
        //    p2.x() = x1;
        //}

        //if (y1 > y2)
        //{
        //    p1.y() = y2;
        //    p2.y() = y1;
        //}

        //dlib::rectangle rect(x1, y1, x2, y2);
        ////dlib::rectangle rect(p1, p2);

        //dlib::matrix<uint8_t> sm = dlib::subm(map, rect);

        //cv::Mat labelImage(cv::Size(sm.nc(), sm.nr()), CV_32S);
        //int nLabels = connectedComponents(dlib::toMat(sm), labelImage, 8);
        //    
        //    
        //    
        //threshold_to_zero(sm, sm, 250, false);

        //dlib::set_subm(map, rect) = sm;

    }   // end of get_points
};

// ----------------------------------------------------------------------------------------

double eval_net(particle p)
{
    long idx;
    dlib::matrix<uint8_t> map, map2;

    dlib::point starting_point = dlib::point(12, 10);
    uint64_t current_points = 0;
    uint64_t moves_without_points = 0;

    dlib::assign_image(map, color_map);

    vehicle vh1(starting_point, 90.0);

    bool crash = false;

    long l2_size = dlib::layer<2>(c_net).layer_details().get_layer_params().size();
    auto l2_data = dlib::layer<2>(c_net).layer_details().get_layer_params().host();
    dlib::matrix<double> x1 = p.get_x1();

    // copy values into the network
    for (idx = 0; idx < l2_size; ++idx)
        *(l2_data + idx) = (float)x1(0,idx);

    long l3_size = dlib::layer<4>(c_net).layer_details().get_layer_params().size();
    auto l3_data = dlib::layer<4>(c_net).layer_details().get_layer_params().host();
    dlib::matrix<double> x2 = p.get_x2();

    for (idx = 0; idx < l3_size; ++idx)
        *(l3_data + idx) = (float)x2(0, idx);

    long l4_size = dlib::layer<6>(c_net).layer_details().get_layer_params().size();
    auto l4_data = dlib::layer<6>(c_net).layer_details().get_layer_params().host();
    dlib::matrix<double> x3 = p.get_x3();

    for (idx = 0; idx < l4_size; ++idx)
        *(l4_data + idx) = (float)x3(0, idx);
/*
    long l5_size = dlib::layer<8>(c_net).layer_details().get_layer_params().size();
    auto l5_data = dlib::layer<8>(c_net).layer_details().get_layer_params().host();
    dlib::matrix<double> x4 = p.get_x4();

    for (idx = 0; idx < l5_size; ++idx)
        *(l5_data + idx) = (float)x4(0, idx);
*/
    uint64_t movement_count = 0;
    
    while (crash == false)
    {
        //current_points = vh1.points;
        
        vh1.check_for_points(map);
        vh1.get_ranges(map, map2);
        win.clear_overlay();
        win.set_image(map2);

        dlib::matrix<float> m3 = dlib::trans(dlib::mat(vh1.detection_ranges));

        //std::cout << dlib::csv << m3 << std::endl;

        dlib::matrix<float> m2 = c_net(m3);

        vh1.move(m2(0, 0), m2(1, 0));

        std::string title = "I: " + num2str(p.iteration, "%04d") + ", N: " + num2str(p.get_number(), "%03d") + ", B: " + num2str(vh1.heading*180.0/dlib::pi, "%2.4f") + ", L/R: " + num2str(m2(0, 0), "%2.4f/") + num2str(m2(1, 0), "%2.4f") + ", Points: " + num2str(-vh1.points, "%4.0f");
        win.set_title(title);

        if(current_points == vh1.points)
        {
            ++movement_count;
        }
        else
        {
            current_points = vh1.points;
            movement_count = 0;
        }

        crash = vh1.test_for_crash(map);

        if (movement_count > 600)
        {
            std::cout << "Count" << std::endl;
            crash = true;
        }

    }

    std::cout << "Particle Number: " << num2str(p.get_number(), "%03d") << ", Points: " << -vh1.points << std::endl;
    //dlib::sleep(200);
    return -vh1.points;
}

// ----------------------------------------------------------------------------------------



int main(int argc, char** argv)
{
    std::string sdate, stime;

    uint64_t idx=0, jdx=0;

    typedef std::chrono::duration<double> d_sec;
    auto start_time = chrono::system_clock::now();
    auto stop_time = chrono::system_clock::now();
    auto elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

    std::ofstream DataLogStream;
    std::string platform;

    get_platform(platform);
    std::cout << "Platform: " << platform << std::endl;

    if (platform.compare(0, 6, "Laptop") == 0)
    {
        std::cout << "Match!" << std::endl;
    }

    try
    {
        int bp = 0;

        // ----------------------------------------------------------------------------------------


        //dlib::matrix<uint32_t> input(1, 28);
        //input = 11, 10, 9, 8, 8, 8, 8, 9, 10, 11, 14, 18, 29, 80, 75, 26, 16, 12, 10, 8, 8, 7, 7, 7, 7, 8, 8, 10;
        //dlib::matrix<uint32_t> input(1, 7);
        //input = 11, 8, 11, 80, 10, 7, 10;
        dlib::matrix<float, 1, 10> input = dlib::ones_matrix<float>(1, 10);

        dlib::matrix<float> motion(2, 1);
        motion = 1.0, 1.0;

        std::cout << c_net << std::endl;

        double intial_learning_rate = 0.0001;
        dlib::dnn_trainer<car_net_type, dlib::adam> trainer(c_net, dlib::adam(0.0001, 0.9, 0.99), { 0 });
        trainer.set_learning_rate(intial_learning_rate);
        trainer.be_verbose();
        trainer.set_test_iterations_without_progress_threshold(5000);

        std::cout << trainer << std::endl;

        for (idx = 0; idx < 10; ++idx)
        {
            trainer.train_one_step({ input }, { motion });
        }

        //dlib::net_to_xml(c_net, "car_net.xml");
      
        // ----------------------------------------------------------------------------------------
        dlib::load_image(color_map, "../maps/test_map_v2_2.png");

        dlib::pso_options options(100, 3000, 2.4, 2.1, 1.0, 1, 1.0);

        std::cout << "----------------------------------------------------------------------------------------" << std::endl;
        std::cout << options << std::endl;

        dlib::pso<particle> p(options);
        p.set_syncfile("../nets/rl_bot_pso_v1.dat");

        //dlib::matrix<double, 1, 2> x1,x2, v1,v2;
        dlib::matrix<double, 1, fc1_size> x1,v1;
        dlib::matrix<double, 1, fc2_size> x2,v2;
        dlib::matrix<double, 1, fc3_size> x3,v3;
        dlib::matrix<double, 1, fc4_size> x4,v4;

        for (idx = 0; idx < x1.nc(); ++idx)
        {
            x1(0, idx) = 1.0;
            v1(0, idx) = 0.01;
        }

        for (idx = 0; idx < x2.nc(); ++idx)
        {
            x2(0, idx) = 100.0;
            v2(0, idx) = 1.0;
        }

        for (idx = 0; idx < x3.nc(); ++idx)
        {
            x3(0, idx) = 100.0;
            v3(0, idx) = 1.0;
        }

        for (idx = 0; idx < x4.nc(); ++idx)
        {
            x4(0, idx) = 100.0;
            v4(0, idx) = 1.0;
        }

        std::pair<particle, particle> x_lim(particle(-x1,-x2,-x3,-x4), particle(x1,x2,x3,x4));
        std::pair<particle, particle> v_lim(particle(-v1,-v2,-v3,-v4), particle(v1,v2,v3,v4));

        p.init(x_lim, v_lim);

        std::cout << c_net << std::endl;
        
        start_time = chrono::system_clock::now();

        p.run(eval_net);
        stop_time = chrono::system_clock::now();
        elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

        std::cout << "PSO (" << elapsed_time.count() << ")" << std::endl;


        std::string filename = "../nets/gbest.dat";
        dlib::serialize(filename) << p.G;


        std::cout << std::endl << "Ready to run G-Best particle..." << std::endl;
        std::cin.ignore();
        
        bp = 3;
        

        particle g_best;

        //dlib::deserialize("gbest_690.dat") >> g_best;
        dlib::deserialize(filename) >> g_best;

        eval_net(g_best);


        bp = 4;


    }
    catch (std::exception &e)
    {
        std::cout << e.what() << std::endl;
    }

    std::cout << "Press Enter to close" << std::endl;
    std::cin.ignore();

	return 0;

}	// end of main

