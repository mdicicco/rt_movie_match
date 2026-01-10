#!/usr/bin/env python3
"""
Movie Critic Matching System

This system finds the movie critics whose taste most closely matches yours
by asking you to rate key "consequential" movies that best differentiate
critics from each other.

The algorithm:
1. Select movies that are BOTH highly reviewed (data coverage) and 
   polarizing (high variance) - these best differentiate critics
2. Ask user to rate these movies on a 1-10 scale
3. Calculate Pearson correlation between user ratings and each critic
4. Iteratively ask for more ratings to refine the match
5. Return top 3 matching critics

Author: Movie Analysis Script
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# File to save user progress
USER_DATA_FILE = 'user_ratings.json'


def load_data(filepath='good_reviews.pk'):
    """Load the critic-movie rating matrix."""
    data = pd.read_pickle(filepath)
    print(f"Loaded data: {len(data.index)} critics, {len(data.columns)} movies")
    return data


def save_user_data(user_ratings, skipped_movies, filepath=USER_DATA_FILE):
    """
    Save user ratings and skipped movies to a JSON file for later resumption.
    """
    data = {
        'ratings': user_ratings,
        'skipped': list(skipped_movies),
        'saved_at': datetime.now().isoformat(),
        'version': 1
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  💾 Progress saved to {filepath}")


def load_user_data(filepath=USER_DATA_FILE):
    """
    Load previously saved user ratings and skipped movies.
    
    Returns:
        user_ratings: dict or None
        skipped_movies: set or None
        saved_at: datetime string or None
    """
    if not os.path.exists(filepath):
        return None, None, None
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        user_ratings = data.get('ratings', {})
        skipped_movies = set(data.get('skipped', []))
        saved_at = data.get('saved_at', 'unknown')
        
        return user_ratings, skipped_movies, saved_at
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  ⚠️  Could not load saved data: {e}")
        return None, None, None


def get_critic_top_movies(data, critic, top_n=50):
    """
    Get a critic's top N highest-rated movies.
    
    Returns a list of (movie, score) tuples sorted by score descending.
    """
    critic_ratings = data.loc[critic].dropna()
    sorted_ratings = critic_ratings.sort_values(ascending=False)
    
    return [(movie, score) for movie, score in sorted_ratings.head(top_n).items()]


def display_critic_recommendations(similarity_df, data, top_n_critics=3, top_n_movies=10):
    """
    Display top movie recommendations from matched critics.
    
    Also highlights movies that appear in the top 50 of ALL matched critics
    (consensus picks that you're likely to enjoy).
    """
    if similarity_df is None or len(similarity_df) == 0:
        return
    
    top_critics = list(similarity_df.head(top_n_critics)['critic'])
    
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + "  🎯  MOVIE RECOMMENDATIONS FROM YOUR MATCHED CRITICS  🎯".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    # Get top 50 from each critic for consensus analysis
    critic_top_50 = {}
    for critic in top_critics:
        critic_top_50[critic] = set(movie for movie, _ in get_critic_top_movies(data, critic, 50))
    
    # Find movies in top 50 of ALL matched critics
    consensus_movies = critic_top_50[top_critics[0]]
    for critic in top_critics[1:]:
        consensus_movies = consensus_movies & critic_top_50[critic]
    
    # Display consensus picks first if any exist
    if consensus_movies:
        print("\n  ⭐ CONSENSUS PICKS (Top 50 for ALL your matched critics):")
        print("  " + "─"*64)
        
        # Get average score across critics for sorting
        consensus_with_scores = []
        for movie in consensus_movies:
            scores = [data.loc[c, movie] for c in top_critics if pd.notna(data.loc[c, movie])]
            avg_score = np.mean(scores) if scores else 0
            consensus_with_scores.append((movie, avg_score))
        
        consensus_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (movie, avg_score) in enumerate(consensus_with_scores[:15], 1):
            score_display = avg_score * 9 + 1 if avg_score <= 1 else avg_score
            print(f"    ★ {i:2}. {format_movie_name(movie)[:45]:<45} avg: {score_display:.1f}/10")
    
    # Display each critic's top 10
    print("\n  ─" + "─"*66)
    
    for critic in top_critics:
        corr = similarity_df[similarity_df['critic'] == critic]['correlation'].values[0]
        print(f"\n  📽️  Top {top_n_movies} from {critic} (correlation: {corr:+.2f}):")
        print("  " + "─"*64)
        
        top_movies = get_critic_top_movies(data, critic, top_n_movies)
        
        for i, (movie, score) in enumerate(top_movies, 1):
            score_display = score * 9 + 1 if score <= 1 else score
            
            # Mark if it's a consensus pick
            consensus_mark = " ⭐" if movie in consensus_movies else ""
            
            print(f"    {i:2}. {format_movie_name(movie)[:45]:<45} {score_display:.1f}/10{consensus_mark}")
    
    # Summary
    print("\n  " + "─"*66)
    if consensus_movies:
        print(f"  💡 {len(consensus_movies)} movies appear in the top 50 of all {top_n_critics} critics!")
        print(f"     These 'consensus picks' (marked with ⭐) are your best bets.")
    else:
        print(f"  💡 No movies in top 50 of all critics - your matched critics have diverse tastes!")


def calculate_movie_scores(data, min_reviews=75):
    """
    Calculate a "consequentiality" score for each movie.
    
    Movies are consequential if they:
    1. Have high variance in critic scores (polarizing)
    2. Have many reviews (good coverage for matching)
    3. Have a robust spread of opinions (IQR-based)
    
    We filter out movies with anomalously high variance (data quality issues).
    """
    movie_stats = pd.DataFrame({
        'count': data.notna().sum(axis=0),
        'mean': data.mean(axis=0),
        'std': data.std(axis=0),
        'variance': data.var(axis=0),
    })
    
    # Filter to movies with enough reviews
    qualified = movie_stats[movie_stats['count'] >= min_reviews].copy()
    
    # Filter out movies with unrealistic variance (likely data quality issues)
    # Variance > 0.15 on a 0-1 scale is suspicious (would mean avg disagreement > 0.38)
    qualified = qualified[qualified['variance'] <= 0.15].copy()
    
    # Calculate IQR for a robust spread measure
    for movie in qualified.index:
        scores = data[movie].dropna()
        if len(scores) > 5:
            q75, q25 = np.percentile(scores, [75, 25])
            qualified.loc[movie, 'iqr'] = q75 - q25
        else:
            qualified.loc[movie, 'iqr'] = 0
    
    # Normalize scores for combining
    var_norm = qualified['variance'] / qualified['variance'].max()
    iqr_norm = qualified['iqr'] / qualified['iqr'].max() if qualified['iqr'].max() > 0 else 0
    count_norm = np.log(qualified['count']) / np.log(qualified['count'].max())
    
    # Final score: balance polarization (60%) with coverage (40%)
    qualified['final_score'] = (
        var_norm * 0.35 +
        iqr_norm * 0.25 +
        count_norm * 0.40
    )
    
    return qualified.sort_values('final_score', ascending=False)


def format_movie_name(movie_slug):
    """Convert movie slug to readable name."""
    import re
    name = movie_slug
    
    # Handle special prefixes (numeric IDs like 10009151-box)
    if name and name[0].isdigit():
        parts = name.split('-', 1)
        if len(parts) > 1 and parts[0].isdigit():
            name = parts[1]
    
    # Replace underscores and hyphens with spaces
    name = name.replace('_', ' ').replace('-', ' ')
    
    # Remove trailing year patterns like " 2015" or " 2010"
    # But keep important years like "2001" in "2001 A Space Odyssey"
    name = re.sub(r'\s+(19\d{2}|20[0-2]\d)$', '', name)
    
    # Title case with exceptions for common words
    words = name.split()
    small_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
    result = []
    for i, word in enumerate(words):
        if i == 0 or word.lower() not in small_words:
            result.append(word.capitalize())
        else:
            result.append(word.lower())
    
    return ' '.join(result)


def get_movie_iterator(movie_stats, already_seen=None):
    """
    Returns an iterator over movies in order of consequentiality.
    
    Skips movies the user has already rated or skipped.
    """
    if already_seen is None:
        already_seen = set()
    
    for movie in movie_stats.index:
        if movie not in already_seen:
            yield movie


def get_user_rating(movie_name, movie_slug):
    """Get a rating from the user for a single movie."""
    print(f"\n  {format_movie_name(movie_slug)}")
    
    while True:
        try:
            response = input("    Your rating (1-10, or 's' to skip, 'q' to finish): ").strip().lower()
            
            if response == 'q':
                return 'quit'
            elif response == 's':
                return None
            else:
                rating = float(response)
                if 1 <= rating <= 10:
                    # Normalize to 0-1 scale (matching the data)
                    return (rating - 1) / 9
                else:
                    print("    Please enter a number between 1 and 10")
        except ValueError:
            print("    Invalid input. Enter 1-10, 's' to skip, or 'q' to quit")


def collect_user_ratings(movie_iterator, target_count, already_rated=None, already_skipped=None):
    """
    Collect ratings from the user, pulling new movies until we get target_count ratings.
    
    If user skips a movie (hasn't seen it), we move to the next movie in the list.
    Continues until we have target_count actual ratings or run out of movies.
    
    Returns:
        user_ratings: dict of movie -> rating
        skipped_movies: set of movies the user skipped
        quit_early: bool indicating if user pressed 'q'
    """
    print("\n" + "="*60)
    print("RATE THESE MOVIES (1 = hated it, 10 = loved it)")
    print("="*60)
    
    if already_rated is None:
        already_rated = {}
    if already_skipped is None:
        already_skipped = set()
    
    user_ratings = {}
    skipped_movies = set()
    movies_shown = 0
    
    for movie in movie_iterator:
        # Skip movies already rated or skipped
        if movie in already_rated or movie in already_skipped:
            continue
            
        movies_shown += 1
        rating = get_user_rating(movie, movie)
        
        if rating == 'quit':
            return user_ratings, skipped_movies, True
        elif rating is None:
            # User skipped - hasn't seen this movie
            skipped_movies.add(movie)
            print(f"    (Skipped - showing next movie...)")
        else:
            user_ratings[movie] = rating
            
            # Check if we've reached target
            if len(user_ratings) >= target_count:
                break
    
    return user_ratings, skipped_movies, False


def calculate_critic_similarity(user_ratings, data):
    """
    Calculate similarity between user ratings and each critic.
    
    Uses Pearson correlation coefficient, which measures how well
    the ratings move together (both like/dislike the same movies).
    
    Returns a DataFrame of critics sorted by similarity.
    """
    if len(user_ratings) < 3:
        print("Need at least 3 rated movies to calculate similarity.")
        return None
    
    similarities = []
    user_movies = list(user_ratings.keys())
    user_scores = np.array([user_ratings[m] for m in user_movies])
    
    for critic in data.index:
        # Get critic's scores for movies the user rated
        critic_scores = data.loc[critic, user_movies]
        
        # Find movies both user and critic have rated
        valid_mask = ~critic_scores.isna()
        common_movies = valid_mask.sum()
        
        if common_movies >= 3:
            critic_valid = critic_scores[valid_mask].values
            user_valid = user_scores[valid_mask.values]
            
            # Calculate Pearson correlation
            if np.std(critic_valid) > 0 and np.std(user_valid) > 0:
                correlation, p_value = stats.pearsonr(user_valid, critic_valid)
                
                # Also calculate mean absolute error for a different perspective
                mae = np.mean(np.abs(user_valid - critic_valid))
                
                similarities.append({
                    'critic': critic,
                    'correlation': correlation,
                    'p_value': p_value,
                    'common_movies': common_movies,
                    'mae': mae,
                    # Combined score weights correlation higher but penalizes low coverage
                    'confidence_score': correlation * np.log(common_movies + 1) / 3
                })
    
    if not similarities:
        return None
    
    result = pd.DataFrame(similarities)
    result = result.sort_values('correlation', ascending=False)
    
    return result


def display_top_critics(similarity_df, data, user_ratings, top_n=3):
    """Display the top N matching critics with details."""
    if similarity_df is None or len(similarity_df) == 0:
        print("No matching critics found. Try rating more movies.")
        return
    
    user_movies = list(user_ratings.keys())
    
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + f"  🎬  YOUR TOP {top_n} MATCHING MOVIE CRITICS  🎬".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    for rank, (_, row) in enumerate(similarity_df.head(top_n).iterrows(), 1):
        critic = row['critic']
        corr = row['correlation']
        common = int(row['common_movies'])
        mae = row['mae']
        
        # Create visual correlation bar
        bar_length = 20
        filled = int(max(0, corr) * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        # Interpret correlation strength
        if corr >= 0.7:
            strength = "★★★ Excellent Match!"
            emoji = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉"
        elif corr >= 0.5:
            strength = "★★☆ Strong Match"
            emoji = "🥈" if rank == 1 else "🥉"
        elif corr >= 0.3:
            strength = "★☆☆ Good Match"
            emoji = "🥉"
        elif corr >= 0.1:
            strength = "☆☆☆ Slight Match"
            emoji = "📊"
        else:
            strength = "No Clear Match"
            emoji = "❓"
        
        print(f"\n  {emoji} #{rank}: {critic}")
        print(f"     ┌────────────────────────────────────────────────────────┐")
        print(f"     │  Similarity: [{bar}] {corr:+.2f}  ")
        print(f"     │  {strength}")
        print(f"     │  Movies in common: {common}")
        print(f"     │  Avg. rating difference: {mae*10:.1f} pts")
        print(f"     └────────────────────────────────────────────────────────┘")
        
        # Show a comparison of movies
        print(f"     Taste Comparison:")
        comparisons = []
        for movie in user_movies:
            critic_score = data.loc[critic, movie]
            if pd.notna(critic_score):
                user_score = user_ratings[movie] * 9 + 1  # Convert back to 1-10
                critic_display = critic_score * 9 + 1 if critic_score <= 1 else critic_score
                diff = abs(user_score - critic_display)
                comparisons.append((movie, user_score, critic_display, diff))
        
        # Show up to 4 comparisons - 2 similar, 2 different
        comparisons.sort(key=lambda x: x[3])
        shown_similar = 0
        shown_diff = 0
        
        for movie, user_score, critic_display, diff in comparisons:
            if shown_similar < 2 and diff < 2:
                match_icon = "✓" if diff < 1 else "≈"
                print(f"       {match_icon} {format_movie_name(movie)[:32]:<32} You: {user_score:.0f}  Critic: {critic_display:.0f}")
                shown_similar += 1
        
        for movie, user_score, critic_display, diff in reversed(comparisons):
            if shown_diff < 2 and diff >= 2:
                print(f"       ✗ {format_movie_name(movie)[:32]:<32} You: {user_score:.0f}  Critic: {critic_display:.0f}")
                shown_diff += 1
    
    # Summary statistics
    print("\n" + "─"*70)
    print("  ANALYSIS SUMMARY")
    print(f"    • Critics analyzed: {len(similarity_df)}")
    print(f"    • Movies you rated: {len(user_ratings)}")
    avg_corr = similarity_df['correlation'].mean()
    print(f"    • Average correlation with all critics: {avg_corr:.3f}")
    
    # Add confidence note
    if len(user_ratings) < 10:
        print(f"\n  💡 Tip: Rate more movies for more accurate matching!")


def run_interactive_matching(data, initial_movies=10, additional_movies=5, max_rounds=5):
    """
    Run the interactive critic matching process.
    
    1. Check for saved progress and offer to resume
    2. Calculate which movies are most consequential
    3. Ask user to rate initial set of movies (keeps asking until target reached)
    4. Show preliminary matches
    5. Offer to rate more movies for better accuracy
    6. Show final top 3 matches with movie recommendations
    7. Save progress for later resumption
    
    If user skips a movie (hasn't seen it), we move to the next movie in the 
    ranked list rather than circling back.
    """
    print("\n")
    print("  ╔═══════════════════════════════════════════════════════════════╗")
    print("  ║                                                               ║")
    print("  ║        🎬  MOVIE CRITIC MATCHING SYSTEM  🎬                  ║")
    print("  ║                                                               ║")
    print("  ║   Find the professional film critic who shares your taste!   ║")
    print("  ║                                                               ║")
    print("  ╚═══════════════════════════════════════════════════════════════╝")
    
    # Check for saved progress
    saved_ratings, saved_skipped, saved_at = load_user_data()
    all_user_ratings = {}
    all_skipped = set()
    
    if saved_ratings and len(saved_ratings) > 0:
        print(f"\n  📂 Found saved progress from {saved_at}")
        print(f"     {len(saved_ratings)} movies rated, {len(saved_skipped) if saved_skipped else 0} skipped")
        response = input("  Resume from saved progress? (y/n): ").strip().lower()
        
        if response == 'y':
            all_user_ratings = saved_ratings
            all_skipped = saved_skipped if saved_skipped else set()
            print(f"  ✓ Loaded {len(all_user_ratings)} previous ratings")
        else:
            print("  Starting fresh...")
    
    # Calculate movie consequentiality scores
    print("\n  Analyzing movie database to find the most revealing films...")
    movie_stats = calculate_movie_scores(data)
    print(f"  ✓ Found {len(movie_stats)} qualifying movies from {len(data.columns)} total")
    print(f"  ✓ Ready to match against {len(data.index)} professional critics")
    
    # Track round number
    round_num = 1
    similarities = None
    
    # If we have enough saved ratings, show matches immediately
    if len(all_user_ratings) >= 3:
        print("\n  🔄 Calculating critic similarities from saved ratings...")
        similarities = calculate_critic_similarity(all_user_ratings, data)
        display_top_critics(similarities, data, all_user_ratings)
        
        remaining = len(movie_stats) - len(all_user_ratings) - len(all_skipped)
        if remaining > 0:
            print("\n" + "─"*70)
            print(f"  You have {len(all_user_ratings)} ratings. Want to add more for better accuracy?")
            response = input("  Rate more movies? (y/n): ").strip().lower()
            if response != 'y':
                # Show recommendations and save
                display_critic_recommendations(similarities, data)
                save_user_data(all_user_ratings, all_skipped)
                _print_goodbye()
                return all_user_ratings, all_skipped, similarities
    
    while round_num <= max_rounds:
        # Target number of movies to collect this round
        if len(all_user_ratings) == 0:
            target = initial_movies
        else:
            target = additional_movies
        
        if len(all_user_ratings) == 0:
            print(f"\n  ─────────────────────────────────────────────────────────────────")
            print(f"  We need you to rate {target} movies you've seen.")
            print(f"  These are films that critics disagree on the most!")
            print(f"  Rate each movie from 1 (hated it) to 10 (loved it).")
            print(f"  Skip movies you haven't seen - we'll show you another one.")
            print(f"  Press 'q' at any time to finish early.")
            print(f"  ─────────────────────────────────────────────────────────────────")
        else:
            print(f"\n  ─── Round {round_num}: Rate {target} more movies ───")
        
        # Create iterator over available movies
        movie_iter = get_movie_iterator(movie_stats, 
                                         already_seen=set(all_user_ratings.keys()) | all_skipped)
        
        # Collect ratings - will keep asking until target is reached
        new_ratings, new_skipped, quit_early = collect_user_ratings(
            movie_iter,
            target_count=target,
            already_rated=all_user_ratings,
            already_skipped=all_skipped
        )
        
        # Update tracking
        all_user_ratings.update(new_ratings)
        all_skipped.update(new_skipped)
        
        # Save progress after each round
        save_user_data(all_user_ratings, all_skipped)
        
        rated_count = len(all_user_ratings)
        skipped_count = len(all_skipped)
        
        print(f"\n  📊 You've rated {rated_count} movie{'s' if rated_count != 1 else ''} total.")
        if skipped_count > 0:
            print(f"     ({skipped_count} movies skipped)")
        
        if quit_early and rated_count < 3:
            print("  ⚠️  Need at least 3 rated movies for matching.")
            break
        
        if rated_count < 3:
            print("  ⚠️  Please rate at least 3 movies to find matches.")
            # Check if we have more movies available
            remaining = len(movie_stats) - len(all_user_ratings) - len(all_skipped)
            if remaining == 0:
                print("  ❌ No more movies available to rate.")
                break
            continue
        
        # Calculate similarities
        print("  🔄 Calculating critic similarities...")
        similarities = calculate_critic_similarity(all_user_ratings, data)
        
        # Display results
        display_top_critics(similarities, data, all_user_ratings)
        
        if quit_early:
            break
        
        # Check if more movies are available
        remaining = len(movie_stats) - len(all_user_ratings) - len(all_skipped)
        if remaining == 0:
            print("\n  ✓ You've rated all available consequential movies!")
            break
        
        # Ask if user wants to continue
        if round_num < max_rounds:
            print("\n" + "─"*70)
            print(f"  More ratings = better accuracy! ({remaining} movies remaining)")
            response = input("  Rate more movies? (y/n): ").strip().lower()
            if response != 'y':
                break
        
        round_num += 1
    
    # Show movie recommendations from matched critics
    if similarities is not None and len(similarities) > 0:
        display_critic_recommendations(similarities, data)
    
    # Final save
    save_user_data(all_user_ratings, all_skipped)
    
    _print_goodbye()
    
    return all_user_ratings, all_skipped, similarities


def _print_goodbye():
    """Print the goodbye message."""
    print("\n")
    print("  ╔═══════════════════════════════════════════════════════════════╗")
    print("  ║   Thank you for using the Movie Critic Matching System! 🍿   ║")
    print("  ║                                                               ║")
    print("  ║   Your ratings have been saved. Run again to add more or     ║")
    print("  ║   see updated recommendations!                               ║")
    print("  ╚═══════════════════════════════════════════════════════════════╝")
    print()


def analyze_critics(data):
    """
    Analyze the critic data to understand patterns.
    """
    print("\n" + "="*70)
    print("CRITIC DATA ANALYSIS")
    print("="*70)
    
    # Reviews per critic
    reviews_per_critic = data.notna().sum(axis=1)
    print(f"\nReviews per critic:")
    print(f"  Min: {reviews_per_critic.min()}")
    print(f"  Max: {reviews_per_critic.max()}")
    print(f"  Mean: {reviews_per_critic.mean():.1f}")
    print(f"  Median: {reviews_per_critic.median():.1f}")
    
    # Most prolific critics
    print(f"\nTop 10 most prolific critics:")
    for critic, count in reviews_per_critic.sort_values(ascending=False).head(10).items():
        print(f"  {count:4.0f} reviews: {critic}")
    
    # Critic rating tendencies (harsh vs generous)
    mean_ratings = data.mean(axis=1)
    print(f"\nCritic rating tendencies:")
    print(f"  Harshest (lowest avg): {mean_ratings.idxmin()} ({mean_ratings.min()*10:.1f}/10)")
    print(f"  Most generous (highest avg): {mean_ratings.idxmax()} ({mean_ratings.max()*10:.1f}/10)")
    
    return reviews_per_critic, mean_ratings


def main():
    """Main entry point for the matching system."""
    # Load data
    data = load_data()
    
    # Optional: Show data analysis
    response = input("\nWould you like to see critic data analysis first? (y/n): ").strip().lower()
    if response == 'y':
        analyze_critics(data)
    
    # Run interactive matching
    ratings, skipped, similarities = run_interactive_matching(
        data, 
        initial_movies=10,
        additional_movies=5,
        max_rounds=5
    )
    
    return ratings, skipped, similarities


if __name__ == '__main__':
    main()
