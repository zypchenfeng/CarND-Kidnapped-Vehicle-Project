/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 121; // set to an odd number matrix for easier tracking
	
	normal_distribution<double> dist_x(0, std[0]);
	normal_distribution<double> dist_y(0, std[1]);
	normal_distribution<double> dist_theta(0, std[2]);
	for (int i = 0; i < num_particles; ++i) {
		Particle new_particle;
		new_particle.id = i;
		new_particle.x = x + dist_x(gen);
		new_particle.y = y + dist_y(gen);
		new_particle.theta = theta + dist_theta(gen);
		new_particle.weight = 1.0;

		particles.push_back(new_particle);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);
	if(fabs(yaw_rate) > 0.00001){
		for (int i = 0; i < num_particles; ++i) {
			particles[i].x += dist_x(gen) + velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += dist_y(gen) + velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += dist_theta(gen) + yaw_rate * delta_t;
		}
	}
	else {
		for (int i = 0; i < num_particles; ++i) {
			particles[i].x += dist_x(gen) + velocity * delta_t * cos(particles[i].theta);
			particles[i].y += dist_y(gen) + velocity * delta_t * sin(particles[i].theta);
			particles[i].theta += dist_theta(gen) + yaw_rate * delta_t;
		}
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	
	for (unsigned int i = 0; i < observations.size(); ++i) {
		double dist_min = std::numeric_limits<double>::infinity();
		int min_index = -1;
		for (unsigned int j = 0; j<predicted.size(); ++j) {
			double dist_new = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
			if (dist_new < dist_min) {
				dist_min = dist_new;
				min_index = predicted[j].id;
			}
		}
		observations[i].id = min_index;
	}
	
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	for(int i=0; i<num_particles; ++i){
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double theta = particles[i].theta;
		
		// filter map_landmarks within sensor range
		vector<LandmarkObs> predictions;
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
			double x_lm = map_landmarks.landmark_list[j].x_f;
			double y_lm = map_landmarks.landmark_list[j].y_f;
			unsigned int id_lm = map_landmarks.landmark_list[j].id_i;
			if (abs(x_lm - p_x) <= sensor_range && abs(y_lm - p_y) <= sensor_range)
				predictions.push_back(LandmarkObs{ id_lm, x_lm, y_lm });			
		}

		// transform observations from particle coordinate to map coordinate
		vector<LandmarkObs> observ_tr;
		for (unsigned int k = 0; k < observations.size(); ++k) {
			double x_tr = p_x + observations[k].x*cos(theta) - observations[k].y*sin(theta);
			double y_tr = p_y + observations[k].x*sin(theta) + observations[k].y*cos(theta);
			observ_tr.push_back(LandmarkObs{ observations[k].id, x_tr, y_tr});
		}

		// associate the predicted lm_map locations with transformed landmark locations
		dataAssociation(predictions, observ_tr);

		// weight observations to landmarks
		particles[i].weight = 1;
		for (unsigned int m = 0; m < observ_tr.size(); ++m) {
			double o_x = observ_tr[m].x;
			double o_y = observ_tr[m].y;
			for (unsigned int n = 0; n < predictions.size(); ++n) {				
				if (predictions[n].id == observ_tr[m].id) {
					double pd_x = predictions[n].x;
					double pd_y = predictions[n].y;
					double std_x = std_landmark[0];
					double std_y = std_landmark[1];
					double w = 1 / (2 * M_PI*std_x * std_y) * exp(-(((pd_x - o_x)*(pd_x - o_x)) / (2.0 * std_x*std_x) + ((pd_y - o_y)*(pd_y - o_y)/ (2.0 * std_y*std_y))));
					particles[i].weight *= w;
				}					
			}			 
		}
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// implement the weight wheel 
	// 1. get the max weight
	
	vector<double> weights;
	for (int i = 0; i <num_particles; ++i) {
		weights.push_back(particles[i].weight);
	}
	double max_weight = *max_element(weights.begin(), weights.end());

	// 2. get a random starting index number
	discrete_distribution<int> rand_int(weights.begin(), weights.end());
	int index = rand_int(gen);
	double d_index = index;

	vector<Particle> new_particles;

	double turn = 0.0;
	for (int i = 0; i <num_particles; ++i) {
		turn += d_index / num_particles * 2.0 * max_weight;
		while (turn > weights[index]) {
			turn -= weights[index];
			index = (index + 1) % num_particles;
		}
		new_particles.push_back(particles[index]);
	}		
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
