#!/usr/bin/env python3
"""
Twitter Alternative Scraper for Profitable Algorithmic Trading Strategies
Uses web scraping to find trading strategy discussions from multiple sources
"""

import requests
import pandas as pd
import re
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import sys
import os
from bs4 import BeautifulSoup
import urllib.parse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingStrategyScraper:
    def __init__(self):
        self.search_keywords = [
            # Algorithmic/Systematic keywords
            "algorithmic trading strategy profitable",
            "systematic trading strategy returns", 
            "quantitative trading strategy profit",
            "algo trading strategy performance",
            "automated trading strategy results",
            
            # Strategy types
            "momentum trading strategy profitable",
            "mean reversion strategy returns",
            "breakout strategy profit", 
            "trend following strategy performance",
            "statistical arbitrage profitable",
            
            # Performance indicators
            "trading strategy profit",
            "trading strategy returns",
            "trading strategy backtest", 
            "trading strategy sharpe",
            
            # Specific mentions
            "trading bot profitable",
            "quant strategy performance",
            "algorithmic returns",
            "systematic profit"
        ]
        
        # Alternative sources for trading strategy content
        self.sources = {
            'reddit': {
                'base_url': 'https://www.reddit.com',
                'search_paths': [
                    '/r/algotrading/search?q={}&restrict_sr=1&sort=relevance',
                    '/r/SecurityAnalysis/search?q={}&restrict_sr=1&sort=relevance',
                    '/r/investing/search?q={}&restrict_sr=1&sort=relevance',
                    '/r/quantfinance/search?q={}&restrict_sr=1&sort=relevance'
                ]
            },
            'nitter': {
                'base_url': 'https://nitter.net',
                'search_path': '/search?f=tweets&q={}'
            }
        }
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        self.profit_indicators = [
            r'\+\d+%', r'\+\d+\.\d+%',  # +X% format
            r'profit', r'profitable', r'gains',
            r'returns?', r'performance',
            r'sharpe\s*ratio', r'sharpe',
            r'backtest', r'backtested',
            r'alpha', r'outperform',
            r'\$\d+', r'\$\d+k', r'\$\d+m',  # Dollar amounts
            r'roi', r'return on investment'
        ]
        
        self.strategy_types = [
            r'algorithmic', r'systematic', r'quantitative',
            r'momentum', r'mean\s*reversion', r'breakout',
            r'trend\s*following', r'arbitrage',
            r'machine\s*learning', r'ml', r'ai',
            r'statistical', r'factor', r'pairs'
        ]
        
    def scrape_nitter(self, search_term: str, max_results: int = 50) -> List[Dict]:
        """Scrape from Nitter (Twitter mirror) for trading strategy content"""
        posts_data = []
        
        try:
            # URL encode the search term
            encoded_term = urllib.parse.quote_plus(search_term)
            search_url = f"{self.sources['nitter']['base_url']}{self.sources['nitter']['search_path'].format(encoded_term)}"
            
            logger.info(f"Scraping Nitter: {search_url}")
            
            response = requests.get(search_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            tweets = soup.find_all('div', class_='timeline-item')
            
            for i, tweet in enumerate(tweets[:max_results]):
                if i >= max_results:
                    break
                    
                try:
                    # Extract tweet content
                    content_elem = tweet.find('div', class_='tweet-content')
                    content = content_elem.get_text().strip() if content_elem else ''
                    
                    # Extract user info
                    user_elem = tweet.find('a', class_='username')
                    username = user_elem.get_text().strip() if user_elem else 'unknown'
                    
                    # Extract stats
                    stats = tweet.find('div', class_='tweet-stats')
                    retweets = 0
                    likes = 0
                    replies = 0
                    
                    if stats:
                        stat_items = stats.find_all('span', class_='tweet-stat')
                        for stat in stat_items:
                            stat_text = stat.get_text().strip()
                            if 'retweet' in stat_text.lower():
                                retweets = self.extract_number(stat_text)
                            elif 'like' in stat_text.lower() or 'heart' in stat_text.lower():
                                likes = self.extract_number(stat_text)
                            elif 'reply' in stat_text.lower():
                                replies = self.extract_number(stat_text)
                    
                    # Extract date
                    date_elem = tweet.find('span', class_='tweet-date')
                    date = date_elem.get('title') if date_elem else datetime.now().isoformat()
                    
                    post_data = {
                        'id': f"nitter_{i}_{hash(content)}",
                        'url': search_url,
                        'date': date,
                        'content': content,
                        'user': username.replace('@', ''),
                        'user_followers': 0,  # Not available on Nitter
                        'user_verified': False,
                        'retweet_count': retweets,
                        'like_count': likes,
                        'reply_count': replies,
                        'quote_count': 0,
                        'search_term': search_term,
                        'source': 'nitter'
                    }
                    
                    posts_data.append(post_data)
                    
                except Exception as e:
                    logger.warning(f"Error parsing tweet {i}: {e}")
                    continue
            
            logger.info(f"Found {len(posts_data)} posts from Nitter for '{search_term}'")
            
        except Exception as e:
            logger.error(f"Error scraping Nitter for '{search_term}': {str(e)}")
        
        return posts_data
    
    def scrape_reddit(self, search_term: str, max_results: int = 50) -> List[Dict]:
        """Scrape Reddit for trading strategy discussions"""
        posts_data = []
        
        for subreddit_path in self.sources['reddit']['search_paths']:
            try:
                # URL encode the search term
                encoded_term = urllib.parse.quote_plus(search_term)
                search_url = f"{self.sources['reddit']['base_url']}{subreddit_path.format(encoded_term)}"
                
                logger.info(f"Scraping Reddit: {search_url}")
                
                response = requests.get(search_url, headers=self.headers, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for post titles and content
                posts = soup.find_all('div', attrs={'data-testid': 'post-container'})
                
                for i, post in enumerate(posts[:max_results//4]):  # Distribute across subreddits
                    try:
                        # Extract title
                        title_elem = post.find('h3')
                        title = title_elem.get_text().strip() if title_elem else ''
                        
                        # Extract post content preview
                        content_elem = post.find('div', attrs={'data-testid': 'post-content'})
                        content = content_elem.get_text().strip()[:500] if content_elem else ''
                        
                        # Combine title and content
                        full_content = f"{title}\n{content}".strip()
                        
                        # Extract user
                        user_elem = post.find('span', string=re.compile(r'u/'))
                        username = user_elem.get_text().strip() if user_elem else 'unknown'
                        
                        # Extract upvotes
                        vote_elem = post.find('div', attrs={'data-testid': 'vote-arrows'})
                        upvotes = 0
                        if vote_elem:
                            vote_text = vote_elem.get_text().strip()
                            upvotes = self.extract_number(vote_text)
                        
                        post_data = {
                            'id': f"reddit_{hash(full_content)}",
                            'url': search_url,
                            'date': datetime.now().isoformat(),
                            'content': full_content,
                            'user': username.replace('u/', ''),
                            'user_followers': 0,
                            'user_verified': False,
                            'retweet_count': 0,
                            'like_count': upvotes,
                            'reply_count': 0,
                            'quote_count': 0,
                            'search_term': search_term,
                            'source': 'reddit'
                        }
                        
                        posts_data.append(post_data)
                        
                    except Exception as e:
                        logger.warning(f"Error parsing Reddit post {i}: {e}")
                        continue
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error scraping Reddit subreddit {subreddit_path}: {str(e)}")
                continue
        
        logger.info(f"Found {len(posts_data)} posts from Reddit for '{search_term}'")
        return posts_data
    
    def extract_number(self, text: str) -> int:
        """Extract number from text string"""
        numbers = re.findall(r'\d+', text)
        return int(numbers[0]) if numbers else 0
    
    def scrape_posts(self, search_term: str, max_results: int = 100) -> List[Dict]:
        """Scrape posts from multiple sources"""
        all_posts = []
        
        # Try Nitter first
        nitter_posts = self.scrape_nitter(search_term, max_results//2)
        all_posts.extend(nitter_posts)
        
        # Then Reddit
        reddit_posts = self.scrape_reddit(search_term, max_results//2)
        all_posts.extend(reddit_posts)
        
        return all_posts
    
    def filter_relevant_tweets(self, tweets: List[Dict]) -> List[Dict]:
        """Filter tweets for relevance to profitable trading strategies"""
        relevant_tweets = []
        
        for tweet in tweets:
            content = tweet['content'].lower()
            
            # Check for profit indicators
            has_profit_indicator = any(
                re.search(indicator, content, re.IGNORECASE) 
                for indicator in self.profit_indicators
            )
            
            # Check for strategy type mentions
            has_strategy_type = any(
                re.search(strategy, content, re.IGNORECASE) 
                for strategy in self.strategy_types
            )
            
            # Calculate relevance score
            relevance_score = 0
            
            # Engagement metrics
            engagement = (tweet['like_count'] + tweet['retweet_count'] + 
                         tweet['reply_count'] + tweet['quote_count'])
            if engagement > 10:
                relevance_score += 1
            if engagement > 50:
                relevance_score += 1
            if engagement > 100:
                relevance_score += 2
                
            # User credibility
            if tweet['user_verified']:
                relevance_score += 2
            if tweet['user_followers'] > 1000:
                relevance_score += 1
            if tweet['user_followers'] > 10000:
                relevance_score += 1
                
            # Content relevance
            if has_profit_indicator:
                relevance_score += 2
            if has_strategy_type:
                relevance_score += 2
                
            # Extract potential performance numbers
            performance_numbers = re.findall(r'\+?\d+\.?\d*%', content)
            if performance_numbers:
                relevance_score += 3
                tweet['performance_numbers'] = performance_numbers
            
            # Add relevance score to tweet
            tweet['relevance_score'] = relevance_score
            
            # Only keep tweets with decent relevance
            if relevance_score >= 3 and (has_profit_indicator or has_strategy_type):
                relevant_tweets.append(tweet)
        
        return relevant_tweets
    
    def extract_strategy_details(self, tweets: List[Dict]) -> List[Dict]:
        """Extract specific strategy details from tweets"""
        for tweet in tweets:
            content = tweet['content']
            
            # Extract strategy mentions
            strategies_mentioned = []
            for strategy in self.strategy_types:
                if re.search(strategy, content, re.IGNORECASE):
                    strategies_mentioned.append(strategy)
            tweet['strategies_mentioned'] = strategies_mentioned
            
            # Extract timeframes
            timeframes = re.findall(r'\b\d+[hmd]\b|\b\d+\s*(?:minute|hour|day|week|month)s?\b', 
                                  content, re.IGNORECASE)
            tweet['timeframes'] = timeframes
            
            # Extract instruments/markets
            instruments = re.findall(r'\b(?:BTC|ETH|SPY|QQQ|ES|NQ|CL|GC|AAPL|TSLA|forex|crypto|stocks?|futures?|options?)\b', 
                                   content, re.IGNORECASE)
            tweet['instruments'] = list(set(instruments))
            
            # Extract backtest periods
            backtest_periods = re.findall(r'(?:backtest|tested|test).*?(?:\d{4}|\d+\s*(?:year|month|day)s?)', 
                                        content, re.IGNORECASE)
            tweet['backtest_periods'] = backtest_periods
            
            # Extract Sharpe ratio mentions
            sharpe_ratios = re.findall(r'sharpe\s*(?:ratio)?\s*[:\-]?\s*(\d+\.?\d*)', 
                                     content, re.IGNORECASE)
            tweet['sharpe_ratios'] = sharpe_ratios
            
        return tweets
    
    def scrape_all_strategies(self, 
                            max_posts_per_term: int = 50) -> pd.DataFrame:
        """Scrape all search terms and compile results"""
        all_posts = []
        
        for i, search_term in enumerate(self.search_keywords):
            logger.info(f"Processing search term {i+1}/{len(self.search_keywords)}: {search_term}")
            
            posts = self.scrape_posts(search_term, max_posts_per_term)
            all_posts.extend(posts)
            
            # Rate limiting - pause between searches
            time.sleep(3)
        
        # Remove duplicates based on content hash
        unique_posts = {post['id']: post for post in all_posts}.values()
        unique_posts = list(unique_posts)
        
        logger.info(f"Total unique posts found: {len(unique_posts)}")
        
        # Filter for relevance
        relevant_posts = self.filter_relevant_tweets(unique_posts)
        logger.info(f"Relevant posts after filtering: {len(relevant_posts)}")
        
        # Extract strategy details
        detailed_posts = self.extract_strategy_details(relevant_posts)
        
        # Convert to DataFrame
        df = pd.DataFrame(detailed_posts)
        
        # Sort by relevance score and engagement
        if not df.empty:
            df['engagement_total'] = (df['like_count'] + df['retweet_count'] + 
                                    df['reply_count'] + df['quote_count'])
            df = df.sort_values(['relevance_score', 'engagement_total'], ascending=False)
        
        return df
    
    def save_results(self, df: pd.DataFrame, filename_prefix: str = "trading_strategies"):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to CSV
        csv_filename = f"{filename_prefix}_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        logger.info(f"Results saved to {csv_filename}")
        
        # Save to JSON for detailed data
        json_filename = f"{filename_prefix}_{timestamp}.json"
        df.to_json(json_filename, orient='records', indent=2, date_format='iso')
        logger.info(f"Detailed results saved to {json_filename}")
        
        # Create summary report
        self.create_summary_report(df, f"{filename_prefix}_summary_{timestamp}.txt")
        
        return csv_filename, json_filename
    
    def create_summary_report(self, df: pd.DataFrame, filename: str):
        """Create a summary report of findings"""
        with open(filename, 'w') as f:
            f.write("TRADING STRATEGY SCRAPING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total tweets analyzed: {len(df)}\n")
            f.write(f"Scraping date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if len(df) > 0:
                f.write("TOP PERFORMING TWEETS (by relevance score):\n")
                f.write("-" * 40 + "\n")
                
                top_tweets = df.head(10)
                for idx, tweet in top_tweets.iterrows():
                    f.write(f"\nRelevance Score: {tweet['relevance_score']}\n")
                    f.write(f"User: @{tweet['user']} (Followers: {tweet['user_followers']})\n")
                    f.write(f"Engagement: {tweet.get('engagement_total', 0)}\n")
                    f.write(f"Content: {tweet['content'][:200]}...\n")
                    f.write(f"URL: {tweet['url']}\n")
                    if 'performance_numbers' in tweet:
                        f.write(f"Performance Numbers: {tweet['performance_numbers']}\n")
                    f.write("-" * 40 + "\n")
                
                # Strategy type analysis
                all_strategies = []
                for strategies in df['strategies_mentioned']:
                    all_strategies.extend(strategies)
                
                if all_strategies:
                    strategy_counts = pd.Series(all_strategies).value_counts()
                    f.write(f"\nMOST MENTIONED STRATEGY TYPES:\n")
                    f.write("-" * 30 + "\n")
                    for strategy, count in strategy_counts.head(10).items():
                        f.write(f"{strategy}: {count} mentions\n")
                
                # Instrument analysis
                all_instruments = []
                for instruments in df['instruments']:
                    all_instruments.extend(instruments)
                
                if all_instruments:
                    instrument_counts = pd.Series(all_instruments).value_counts()
                    f.write(f"\nMOST MENTIONED INSTRUMENTS:\n")
                    f.write("-" * 30 + "\n")
                    for instrument, count in instrument_counts.head(10).items():
                        f.write(f"{instrument}: {count} mentions\n")
        
        logger.info(f"Summary report saved to {filename}")


def main():
    """Main execution function"""
    scraper = TradingStrategyScraper()
    
    print("Starting multi-source scrape for profitable algorithmic trading strategies...")
    print("Scraping from Nitter (Twitter mirror) and Reddit...")
    print("This may take several minutes depending on the number of search terms and rate limits.")
    
    # Scrape posts
    df = scraper.scrape_all_strategies(max_posts_per_term=30)
    
    if df.empty:
        print("No relevant posts found. Try adjusting search terms or check internet connection.")
        return
    
    print(f"\nFound {len(df)} relevant posts!")
    
    # Show source breakdown
    if 'source' in df.columns:
        source_counts = df['source'].value_counts()
        print(f"Source breakdown:")
        for source, count in source_counts.items():
            print(f"- {source}: {count} posts")
    
    # Save results
    csv_file, json_file = scraper.save_results(df)
    
    print(f"\nResults saved:")
    print(f"- CSV: {csv_file}")
    print(f"- JSON: {json_file}")
    print(f"- Summary: {csv_file.replace('.csv', '_summary.txt')}")
    
    # Display top results
    print(f"\nTop 5 most relevant posts:")
    print("-" * 50)
    
    for idx, post in df.head(5).iterrows():
        print(f"\nRelevance Score: {post['relevance_score']}")
        print(f"Source: {post.get('source', 'unknown')}")
        print(f"@{post['user']} ({post['user_followers']} followers)")
        print(f"Content: {post['content'][:150]}...")
        print(f"URL: {post['url']}")
        if 'performance_numbers' in post and post['performance_numbers']:
            print(f"Performance: {post['performance_numbers']}")


if __name__ == "__main__":
    main()
