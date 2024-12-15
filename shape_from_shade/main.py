"""Implements the main file for part2."""

import argparse
from shape_from_shading import ShapeFromShading
import time

def main():

  parser = argparse.ArgumentParser(
      description='CS59300CVD Assignment 3 Part 2')
  parser.add_argument('-s', '--subject_name', default='yaleB01',
                      type=str, help="Name of the subject to use.")
  parser.add_argument('-i', '--integration_method', default='random',
                      type=str, help="Integration method to use. Supports ['random', 'average', 'column, row].")
  args = parser.parse_args()

  root_path = 'data/croppedyale/'
  subject_name = args.subject_name
  integration_method = args.integration_method
  full_path = '%s%s' % (root_path, subject_name)

  start_time = time.time()
  ShapeFromShading(full_path, subject_name,
                   integration_method=integration_method)
  end_time = time.time()
  print('Time taken: %f' % (end_time - start_time))

if __name__ == '__main__':
  main()
