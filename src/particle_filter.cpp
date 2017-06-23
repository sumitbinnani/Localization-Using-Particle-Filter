/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iterator>

#include "particle_filter.h"

using namespace std;
const double EPSILON = 1e-6;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 50;
	particles.resize(num_particles);
	weights.resize(num_particles);

	// Normal distribution for x, y and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	default_random_engine random_engine;

	// create particles and set their values
	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = dist_x(random_engine); // take a random value from the Gaussian Normal distribution and update the attribute
		p.y = dist_y(random_engine);
		p.theta = dist_theta(random_engine);
		p.weight = 1;
		particles[i] = p;
		weights[i] = p.weight;
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	normal_distribution<double> x_normal(0, std_pos[0]);
	normal_distribution<double> y_normal(0, std_pos[1]);
	normal_distribution<double> theta_normal(0, std_pos[2]);

	default_random_engine random_engine;

	// Add measurements to each particle and add random Gaussian noise.
	for (int i = 0; i < num_particles; i++) {
		Particle *p = &particles[i];
		if (fabs(yaw_rate) < EPSILON) {
			p->x += velocity * delta_t * cos(p->theta);
			p->y += velocity * delta_t * sin(p->theta);
		} else {
			double r = velocity / yaw_rate;
			double new_theta = p->theta + yaw_rate * delta_t;
			p->x += r * (sin(new_theta) - sin(p->theta));
			p->y += r * (cos(p->theta) - cos(new_theta));
			p->theta = new_theta;
		}
		p->x += x_normal(random_engine);
		p->y += y_normal(random_engine);
		p->theta += theta_normal(random_engine);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	for (auto pred : predicted) {
		double dist_min = numeric_limits<double>::max();
		for (auto observation : observations) {
			double distance = dist(observation.x, observation.y, pred.x, pred.y);
			if (distance < dist_min) {
				observation.id = pred.id;
			}
			dist_min = distance;
		}
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	double weights_sum = 0;

	for (int i = 0; i < num_particles; i++) {
		Particle *p = &particles[i];
		double wt = 1.0;

		// transform vehicle's co-ordinate to map's coordinate system
		for (int j = 0; j < observations.size(); j++) {
			LandmarkObs current_obs = observations[j];
			LandmarkObs transformed_obs;

			transformed_obs.x = (current_obs.x * cos(p->theta)) - (current_obs.y * sin(p->theta)) + p->x;
			transformed_obs.y = (current_obs.x * sin(p->theta)) + (current_obs.y * cos(p->theta)) + p->y;
			transformed_obs.id = current_obs.id;

			// assign predicted measurement that is closest to each observed landmark
			Map::single_landmark_s landmark;
			double distance_min = numeric_limits<double>::max();

			for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
				Map::single_landmark_s cur_l = map_landmarks.landmark_list[k];
				double distance = dist(transformed_obs.x, transformed_obs.y, cur_l.x_f, cur_l.y_f);
				if (distance < distance_min) {
					distance_min = distance;
					landmark = cur_l;
				}
			}

			// update weights using Multivariate Gaussian Distribution
			double num = exp(-0.5 * (pow((transformed_obs.x - landmark.x_f), 2) / pow(std_landmark[0], 2) +
			                         pow((transformed_obs.y - landmark.y_f), 2) / pow(std_landmark[1], 2)));
			double denom = 2 * M_PI * std_landmark[0] * std_landmark[1];
			wt *= num / denom;
		}
		weights_sum += wt;
		p->weight = wt;
	}

	// normalize weights
	for (int i = 0; i < num_particles; i++) {
		Particle *p = &particles[i];
		p->weight /= weights_sum;
		weights[i] = p->weight;
	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine random_engine;

	// the probability of each paticle is propotional to its weight
	discrete_distribution<int> distribution(weights.begin(), weights.end());
	vector<Particle> resampled_particles;

	for (int i = 0; i < num_particles; i++) {
		resampled_particles.push_back(particles[distribution(random_engine)]);
	}

	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best) {
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}

string ParticleFilter::getSenseX(Particle best) {
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}

string ParticleFilter::getSenseY(Particle best) {
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
