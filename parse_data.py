import json
import pandas as pd

critics = {}
movies = {}

raw_data = json.load(open('rt_output.json', 'r'))

error_count = 0

for entry in raw_data:
    critic = entry.keys()[0]
    review_data = entry[critic]

    try:
        _, review_type, movie_name = review_data[2].split('/')[0:3]

        review_number = review_data[3]

        if review_number == '':
            # No number, just a fresh/rotten, ignore these
            continue
        elif review_type == 't':
            # It's a TV show, not a movie, ignore
            continue
        else:
            # Normal critic
            if critic in critics:
                critics[critic][0] += 1
                critics[critic][1].append(movie_name)
            else:
                critics[critic] = [1, [movie_name]]

            if movie_name in movies:
                movies[movie_name][0] += 1
                movies[movie_name][1].append(critic)
            else:
                movies[movie_name] = [1, [critic]]

    except ValueError:
        error_count += 1
        print error_count, "| error:", entry

print "Initial:", len(movies), "movies reviewed by", len(critics), "critics"

movie_review_limit = 75
critic_review_limit = 55

sub_critics = set()
sub_movies = set()

for movie, moviedata in movies.iteritems():
    num_reviews = moviedata[0]
    critic_list = moviedata[1]

    if num_reviews >= movie_review_limit:
        for critic in critic_list:
            if critics[critic][0] >= critic_review_limit:
                sub_critics.add(critic)
        sub_movies.add(movie)

print "Final:", len(sub_movies), "movies reviewed by", len(sub_critics), "critics"




# 75, 55 => 2383 movies, 987 critics => 2,352,021
#100, 75 => 1495 movies, 891 critics => 1,332,045
#100,100 => 1495 movies, 798 critics => 1,193,010

# Drop the binary reviews:
# 75, 50 => 1619 movies, 782 critics => 1,266,058
# 75, 55 => 1619 movies, 753 critics => 1,219,107


letters = ['A+', 'A', 'A-',
           'B+', 'B', 'B-',
           'C+', 'C', 'C-',
           'D+', 'D', 'D-',
           'F+', 'F', 'F-']
grades = [5.0, 4.583333333333334, 4.166666666666667,
          3.75, 3.3333333333333335, 2.916666666666667,
          2.5, 2.0833333333333335, 1.6666666666666667,
          1.25, 0.8333333333333334, 0.4166666666666667,
          0.0, 0.0, 0.0]
grade_map = {letter:number for letter, number in zip(letters, grades)}

#low +1 out of -4..+4

def convert_to_number(review):

    if 'out of -4..+4' in review:
        # Special case of one weird reviewer
        raise ValueError

    if '/' in review:
        # If it has a fraction in it, use that to calculate
        # Examples: 3/4, 3.5/5, or 7.2/10
        parts = review.split('/')
        if float(parts[0]) == 0.0:
            return 0.0
        print 'Found number match:', parts[0], parts[1], 'Result:', float(parts[0])/float(parts[1])
        return float(parts[0])/float(parts[1])

    elif str(review) in grades:
        # Assume it's a letter grade
        print 'Found letter:', review, grade_maps[review], 'Result:', grade_maps[review]/5.0
        return grade_maps[review]/5.0

    else:
        # Assume it's just a number, if so, it's probably out of 4
        # Use a try block just in case, that should get all garbage
        # data, which hopefully will be minimal
        if type(review) in [int, float]:
            print 'Lonely letter: ', review, 'Result:', float(review)/4.0
            return float(review)/4.0

    raise ValueError

# Create the pandas data frame
indexes = list(sub_critics)
columns = list(sub_movies)

data = pd.DataFrame(index=indexes, columns=columns)

for entry in raw_data:
    critic = entry.keys()[0]
    review_data = entry[critic]

    try:
        _, review_type, movie_name = review_data[2].split('/')[0:3]

        if critic in sub_critics and movie_name in sub_movies:
            review_number = convert_to_number(review_data[3])
            data[movie_name][critic] = review_number

    except ValueError:
        continue

data.to_pickle('good_reviews.csv')
