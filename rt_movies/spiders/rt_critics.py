# -*- coding: utf-8 -*-
import scrapy

DEBUG_PRINT = False

def debug_print(*args):
    if DEBUG_PRINT:
        print args

class RtCriticsSpider(scrapy.Spider):
    name = "rt_critics"
    allowed_domains = ["rottentomatoes.com"]
    start_urls = ['https://www.rottentomatoes.com/critics/authors']

    # This is the entry point, which is why the name is just "parse"
    # The start URL is the 'authors' page which has many sub pages for each letter
    # The sub pages will be parsed to get the individual critic pages
    def parse(self, response):
        # Start on the main reviewers page and find the links to all the letter
        # pages, each of which will have reviewers by name
        header = response.css("ul.alpha-pills")
        for letter in header.css('li'):
            url = letter.css("::attr(href)").extract_first()
            debug_print("Branching to parse URL:", url)
            yield scrapy.Request(response.urljoin(url), self.parse_letter_link)


    def parse_letter_link(self, response):
        # From the main critics 'letter' page, find all the critic names
        # Run the 'find number of pages of reviews' parser on that page
        # This will then part through each page and find the reviews
        for critic_selector in response.css("p.critic-names"):
            item = {}
            critic = critic_selector.select("a")
            item['name'] = critic.css("::text").extract_first()
            url = item['url'] = critic.css("::attr(href)").extract_first()
            debug_print("Found a critic:", item)
            debug_print("Branching to parse URL:", url)
            yield scrapy.Request(response.urljoin(url), self.find_number_of_review_pages)


    def find_number_of_review_pages(self, response):
        # Each critic will have multiple pages of reviews, so the first task is to scrape
        # the number of pages.  After that, the actual movies can be extracted from
        # the various pages because the page html has high level summaries
        # Example URL for the first critic found with more than 1 page:
        # https://www.rottentomatoes.com/critic/simon-abrams/movies?page=1

        # Get the main table:0
        chart_main = response.css('section[id=criticsReviewsChart_main]')

        # Get the 'li' items from the chart
        # the last one is the link to the final page
        final_page = chart_main.css('li')[-1]

        # Extract the URL for the final page
        final_page_url = final_page.css('::attr(href)').extract()

        if final_page_url == [u'#']:
            # less than 1 page, just parse this page
            debug_print("Critic has one page:", response.url)
            yield scrapy.Request(response.url, self.parse_movie_reviews)
            pass
        else:
            # Multiple pages, find out how many and run parser on each page

            # This URL should look like '?page=58'
            # example: [u'?page=58']
            # extract the number 58 then make a generator
            # that creates all of the page URLs
            page_text = final_page_url[0]
            max_page_number = int(page_text[6:])
            page_prefix = page_text[0:6]

            # Iterate through all pages for this critic and call the parse_move_reviews
            # funciton
            for i in range(max_page_number):
                url = page_prefix + str(i+1)
                debug_print("Extracting critic page number, found: ", max_page_number, " working on:", i)
                debug_print("Critic has multiple pages page:", response.urljoin(url))
                yield scrapy.Request(response.urljoin(url), self.parse_movie_reviews)


    def parse_movie_reviews(self, response):
        # Go through all the reviews on thsi one page
        # Find all the movies, extract the scores
        # return type should be critic, movie, score

        # Get the critic name from the 'criticsSidebar_main' section
        sidebar = response.css('section[id=criticsSidebar_main]')
        reviewer = sidebar.css('h2').css('::text').extract_first().strip()

        # Get the main table:
        chart_main = response.css('section[id=criticsReviewsChart_main]')

        # This should be a table with 51 elements: 1 header and 50 movies
        for table_row in chart_main.css('tr'):
            row_text = table_row.css('::text').extract()

            # First row is headers so it has less elements (11 vs 23)
            if len(row_text) > 11:

                debug_print("RAW ROW TEXT:", row_text)

                table_elements = table_row.css('td')
                rating_element = table_elements[0]
                rating_data = rating_element.css('span').extract()[0]

                freshness = None
                if 'fresh' in rating_data:
                    freshness = 'FRESH'
                else:
                    freshness = 'ROTTEN'

                rating = row_text[2].strip()
                movie_name = row_text[10]

                # Occasionally movie name is just empty whitespace, just skip
                # This is often an error on the website
                if movie_name.strip() == '':
                    continue

                date = 'unknown'
                for elem in row_text:
                    if "Posted" in elem:
                        # If it has the text "Posted" it's probably the date
                        # if it is the date, remove the first 7 characters
                        # to remove "Posted "
                        date = elem.strip()[7:]

                url = table_row.css('::attr(href)').extract_first()
                try:
                    debug_print('-----')
                    debug_print('Found rating: ')
                    debug_print(movie_name)
                    debug_print(rating)
                    debug_print(freshness)
                    debug_print(date)
                    debug_print(url)
                    debug_print(reviewer)
                    debug_print('-----')
                    debug_print('')
                except UnicodeEncodeError:
                    debug_print('Attempted to print data but got unicode error, continuing')
                yield {reviewer : (movie_name, date, url, rating, freshness)}
