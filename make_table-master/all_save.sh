#!/bin/sh
psql -d testsql -c "copy main to stdout with csv header delimiter ',';" > ../Investigation/bin/table/main.csv

psql -d testsql -c "copy color to stdout with csv header delimiter ',';" > ../Investigation/bin/table/color.csv

psql -d testsql -c "copy size to stdout with csv header delimiter ',';" > ../Investigation/bin/table/size.csv

psql -d testsql -c "copy shape to stdout with csv header delimiter ',';" > ../Investigation/bin/table/shape.csv

psql -d testsql -c "copy taste to stdout with csv header delimiter ',';" > ../Investigation/bin/table/taste.csv

psql -d testsql -c "copy hardness to stdout with csv header delimiter ',';" > ../Investigation/bin/table/hardness.csv

psql -d testsql -c "copy move to stdout with csv header delimiter ',';" > ../Investigation/bin/table/move.csv

psql -d testsql -c "copy name to stdout with csv header delimiter ',';" > ../Investigation/bin/table/name.csv

psql -d testsql -c "copy move_list to stdout with csv header delimiter ',';" > ../Investigation/bin/table/move_list.csv

psql -d testsql -c "copy new_color to stdout with csv header delimiter ',';" > ../Investigation/bin/table/new_color.csv

psql -d testsql -c "copy new_taste to stdout with csv header delimiter ',';" > ../Investigation/bin/table/new_taste.csv

psql -d testsql -c "copy new_hardness to stdout with csv header delimiter ',';" > ../Investigation/bin/table_last/new_hardness.csv

psql -d testsql -c "copy new_size to stdout with csv header delimiter ',';" > ../Investigation/bin/table_last/new_size.csv
