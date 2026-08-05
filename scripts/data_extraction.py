# enhanced_data_extraction.py
import zstandard as zstd
import io
import pandas as pd
import os
import re
import time
import json
from pathlib import Path

def extract_enhanced_games_chunked(file_path, chunk_size=100000, max_games=1000000, output_dir="data/enhanced_extraction"):
    """
    Enhanced extraction with ALL Lichess fields - Small experiment first
    """
    print(f"🔄 ENHANCED Processing from: {file_path}")
    print(f"💾 Target: {max_games:,} games in chunks of {chunk_size:,}")
    print(f"🎯 Experiment mode - extracting ALL available fields")
    print(f"📁 Output dir: {output_dir}")
    print("-" * 60)

    current_chunk = []
    games_processed = 0
    games_filtered = 0  # Track filtered games
    total_kept = 0
    chunk_number = 0
    start_time = time.time()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Enhanced regex patterns for ALL Lichess fields (UPDATED)
    header_patterns = {
        # Game metadata
        'event': re.compile(r'\[Event "([^"]+)"\]'),
        'site': re.compile(r'\[Site "([^"]+)"\]'),
        'date': re.compile(r'\[Date "([^"]+)"\]'),
        'utc_date': re.compile(r'\[UTCDate "([^"]+)"\]'),
        'utc_time': re.compile(r'\[UTCTime "([^"]+)"\]'),
        'annotator': re.compile(r'\[Annotator "([^"]+)"\]'),
        
        # Players & ratings
        'white_player': re.compile(r'\[White "([^"]+)"\]'),
        'black_player': re.compile(r'\[Black "([^"]+)"\]'),
        'white_elo': re.compile(r'\[WhiteElo "(\d+)"\]'),
        'black_elo': re.compile(r'\[BlackElo "(\d+)"\]'),
        'white_rating_diff': re.compile(r'\[WhiteRatingDiff "([^"]+)"\]'),
        'black_rating_diff': re.compile(r'\[BlackRatingDiff "([^"]+)"\]'),
        'white_title': re.compile(r'\[WhiteTitle "([^"]+)"\]'),
        'black_title': re.compile(r'\[BlackTitle "([^"]+)"\]'),
        
        # Game details
        'result': re.compile(r'\[Result "([^"]+)"\]'),
        'variant': re.compile(r'\[Variant "([^"]+)"\]'),
        'time_control': re.compile(r'\[TimeControl "([^"]+)"\]'),
        'termination': re.compile(r'\[Termination "([^"]+)"\]'),
        
        # Opening information
        'eco': re.compile(r'\[ECO "([^"]+)"\]'),
        'opening': re.compile(r'\[Opening "([^"]+)"\]'),
        
        # Lichess-specific fields that might exist
        'round': re.compile(r'\[Round "([^"]+)"\]'),
        'fen': re.compile(r'\[FEN "([^"]+)"\]'),
        'setup': re.compile(r'\[SetUp "([^"]+)"\]'),
    }
    
    # Track field completeness
    field_stats = {field: 0 for field in header_patterns.keys()}
    
    with open(file_path, 'rb') as compressed_file:
        dctx = zstd.ZstdDecompressor()
        
        with dctx.stream_reader(compressed_file) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            
            current_game = {}
            move_buffer = []
            in_moves = False
            
            for line in text_stream:
                line = line.strip()
                
                # Detect start of moves section
                if line and (line[0].isdigit() or line.startswith('1.')):
                    in_moves = True
                    move_buffer.append(line)
                    continue
                
                # End of game (empty line after moves)
                if not line and in_moves:
                    # Process complete game
                    if current_game and 'white_player' in current_game and 'black_player' in current_game:
                        # Add derived features
                        enhanced_game = add_derived_features(current_game, move_buffer)
                        games_processed += 1
                        
                        # FILTER: Only keep rated, non-bullet games
                        white_elo = enhanced_game.get('white_elo', 0) or 0
                        black_elo = enhanced_game.get('black_elo', 0) or 0
                        time_class = enhanced_game.get('time_class', '')
                        
                        # Skip if unrated or bullet
                        if white_elo == 0 or black_elo == 0 or time_class == 'bullet':
                            games_filtered += 1
                            # Reset for next game
                            current_game = {}
                            move_buffer = []
                            in_moves = False
                            continue
                        
                        current_chunk.append(enhanced_game)
                        
                        # Update field statistics
                        for field in enhanced_game:
                            if field in field_stats and enhanced_game[field] is not None and enhanced_game[field] != '':
                                field_stats[field] += 1
                        
                        # Write chunk when full
                        if len(current_chunk) >= chunk_size:
                            write_enhanced_chunk(current_chunk, chunk_number, output_dir, field_stats, games_processed)
                            total_kept += len(current_chunk)
                            current_chunk = []
                            chunk_number += 1
                        
                        # Progress update
                        if games_processed % 5000 == 0:
                            elapsed = time.time() - start_time
                            rate = games_processed / elapsed if elapsed > 0 else 0
                            kept_rate = (total_kept + len(current_chunk)) / games_processed * 100 if games_processed > 0 else 0
                            print(f"📊 Processed: {games_processed:,} | Kept: {total_kept + len(current_chunk):,} ({kept_rate:.1f}%) | Rate: {rate:.0f}/sec")
                        
                        # Stop at max games for experiment
                        if total_kept + len(current_chunk) >= max_games:
                            break
                    
                    # Reset for next game
                    current_game = {}
                    move_buffer = []
                    in_moves = False
                    continue
                
                # Skip move lines when we're in the moves section
                if in_moves:
                    move_buffer.append(line)  # Continue collecting move lines
                    continue
                
                # Parse headers - SPECIAL HANDLING for site to extract game_id
                if line.startswith('[Site '):
                    site_match = header_patterns['site'].match(line)
                    if site_match:
                        site_url = site_match.group(1)
                        current_game['site'] = site_url
                        # Extract game ID from Lichess URL
                        game_id_match = re.search(r'lichess\.org/([A-Za-z0-9]+)', site_url)
                        if game_id_match:
                            current_game['game_id'] = game_id_match.group(1)
                    continue
                
                # Parse other headers
                for field_name, pattern in header_patterns.items():
                    if field_name == 'site':  # Already handled above
                        continue
                    match = pattern.match(line)
                    if match:
                        current_game[field_name] = match.group(1)
                        break
    
    # Write final chunk
    if current_chunk:
        write_enhanced_chunk(current_chunk, chunk_number, output_dir, field_stats, games_processed)
        total_kept += len(current_chunk)
    
    # Final statistics
    total_time = time.time() - start_time
    kept_rate = (total_kept / games_processed) * 100 if games_processed > 0 else 0
    print(f"\n✅ Enhanced Extraction Completed!")
    print(f"   Total games processed: {games_processed:,}")
    print(f"   Valid games kept: {total_kept:,} ({kept_rate:.1f}%)")
    print(f"   Games filtered out: {games_filtered:,}")
    print(f"   Written to {chunk_number + 1} chunk files")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Average rate: {games_processed/total_time:.0f} games/second")
    
    # Combine chunks and analyze
    final_path = combine_enhanced_chunks(output_dir, chunk_number + 1, field_stats)
    analyze_extraction_results(final_path, field_stats, total_time)
    
    return final_path

def add_derived_features(game_data, moves):
    """Add calculated features to game data - FIXED VERSION"""
    enhanced = game_data.copy()
    
    # Basic ELO calculations
    try:
        white_elo = int(game_data.get('white_elo', 0))
        black_elo = int(game_data.get('black_elo', 0))
        enhanced['avg_elo'] = (white_elo + black_elo) / 2
        enhanced['elo_diff'] = abs(white_elo - black_elo)
    except (ValueError, TypeError):
        enhanced['avg_elo'] = None
        enhanced['elo_diff'] = None
    
    # Time control classification
    time_control = game_data.get('time_control', '')
    enhanced['time_class'] = classify_time_control(time_control)
    
    # Move analysis - FIXED
    move_sequence = ' '.join(moves) if moves else ''
    enhanced['move_sequence'] = move_sequence
    enhanced['total_move_lines'] = len(moves)  # Number of move lines, not individual moves
    
    # Calculate actual move count from move sequence
    if move_sequence:
        # Count move numbers (1., 2., 3., etc.) to get actual moves
        move_numbers = len(re.findall(r'\d+\.', move_sequence))
        enhanced['actual_move_count'] = move_numbers/2
        
        # Count chess patterns
        enhanced['captures_count'] = move_sequence.count('x')
        enhanced['checks_count'] = move_sequence.count('+')
        enhanced['castling_count'] = move_sequence.count('O-O')
        enhanced['promotion_count'] = move_sequence.count('=')
        enhanced['has_engine_analysis'] = '[%eval' in move_sequence
        
        # Calculate ply count (half-moves) - approximate
        enhanced['calculated_ply_count'] = move_numbers * 2  # Rough estimate
    else:
        enhanced['actual_move_count'] = 0
        enhanced['captures_count'] = 0
        enhanced['checks_count'] = 0
        enhanced['castling_count'] = 0
        enhanced['promotion_count'] = 0
        enhanced['has_engine_analysis'] = False
        enhanced['calculated_ply_count'] = 0
    
    # Quality indicators - IMPROVED
    enhanced['is_standard_variant'] = game_data.get('variant', 'Standard') == 'Standard'  # Default to Standard if missing
    enhanced['is_rated_game'] = white_elo > 0 and black_elo > 0
    enhanced['min_moves_met'] = enhanced['actual_move_count'] >= 10  # At least 10 full moves
    enhanced['valid_result'] = game_data.get('result', '') in ['1-0', '0-1', '1/2-1/2']
    enhanced['normal_termination'] = game_data.get('termination', '') == 'Normal'
    
    # Overall quality score
    quality_checks = [
        enhanced['is_standard_variant'],
        enhanced['is_rated_game'], 
        enhanced['min_moves_met'],
        enhanced['valid_result'],
        enhanced['time_class'] not in ['bullet', 'unknown'],
        white_elo >= 800 if white_elo else False,
        black_elo >= 800 if black_elo else False
    ]
    enhanced['quality_score'] = sum(bool(check) for check in quality_checks)
    enhanced['is_quality_game'] = enhanced['quality_score'] >= 5
    
    return enhanced

def classify_time_control(time_control_str):
    """Classify time control based on initial time only (Lichess standard)"""
    if not time_control_str or time_control_str == '-':
        return 'unknown'
    
    try:
        # Handle format: "600+0" or "180+2" - extract only the initial time
        if '+' in time_control_str:
            parts = time_control_str.split('+')
            initial_time = int(parts[0])  # Only use base time, ignore increment
        else:
            # Handle format without increment: "600"
            initial_time = int(time_control_str)
        
        # Lichess classification based on initial time only
        if initial_time < 180:      # Less than 3 minutes
            return 'bullet'
        elif initial_time < 600:    # 3-10 minutes  
            return 'blitz'
        elif initial_time < 1800:   # 10-30 minutes
            return 'rapid'
        else:                       # 30+ minutes
            return 'classical'
            
    except (ValueError, IndexError):
        return 'unknown'

def write_enhanced_chunk(chunk_data, chunk_number, output_dir, field_stats, total_processed):
    """Write enhanced chunk with statistics"""
    chunk_df = pd.DataFrame(chunk_data)
    chunk_file = os.path.join(output_dir, f"enhanced_chunk_{chunk_number:03d}.csv")
    chunk_df.to_csv(chunk_file, index=False)
    
    # Calculate file size
    file_size_mb = os.path.getsize(chunk_file) / (1024 * 1024)
    
    print(f"💾 Chunk {chunk_number}: {len(chunk_data):,} games, {file_size_mb:.1f} MB, {len(chunk_df.columns)} columns")

def combine_enhanced_chunks(output_dir, num_chunks, field_stats):
    """Combine chunks and create analysis"""
    print(f"\n🔄 Combining {num_chunks} enhanced chunks...")
    
    all_games = []
    
    for i in range(num_chunks):
        chunk_file = os.path.join(output_dir, f"enhanced_chunk_{i:03d}.csv")
        if os.path.exists(chunk_file):
            chunk_df = pd.read_csv(chunk_file)
            all_games.append(chunk_df)
            # Keep chunk files for now (don't delete during experiment)
    
    if all_games:
        final_df = pd.concat(all_games, ignore_index=True)
        
        # Save with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        final_path = os.path.join(output_dir, f"enhanced_experiment_{timestamp}.csv")
        final_df.to_csv(final_path, index=False)
        
        # Calculate final file size
        file_size_mb = os.path.getsize(final_path) / (1024 * 1024)
        
        print(f"💾 Final dataset: {len(final_df):,} games, {len(final_df.columns)} columns, {file_size_mb:.1f} MB")
        print(f"📄 Saved to: {final_path}")
        
        # Save field completeness stats
        stats_path = os.path.join(output_dir, f"field_stats_{timestamp}.json")
        with open(stats_path, 'w') as f:
            json.dump(field_stats, f, indent=2)
        
        return final_path
    
    return None

def analyze_extraction_results(file_path, field_stats, processing_time):
    """Analyze extraction results - ENHANCED"""
    if not file_path or not os.path.exists(file_path):
        return
    
    df = pd.read_csv(file_path)
    
    print(f"\n📊 EXTRACTION ANALYSIS:")
    print(f"=" * 50)
    print(f"Total games extracted: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"File size: {os.path.getsize(file_path) / (1024*1024):.1f} MB")
    print(f"Processing time: {processing_time/60:.1f} minutes")
    print(f"Processing rate: {len(df)/processing_time:.0f} games/second")
    
    # Field completeness
    print(f"\n📋 Field Completeness:")
    for field, count in field_stats.items():
        percentage = (count / len(df)) * 100 if len(df) > 0 else 0
        print(f"   {field:20}: {count:>6,} ({percentage:5.1f}%)")
    
    # Quality analysis
    if 'is_quality_game' in df.columns:
        quality_games = df['is_quality_game'].sum()
        print(f"\n🎯 Quality Analysis:")
        print(f"   Quality games: {quality_games:,} ({quality_games/len(df)*100:.1f}%)")
        
        if 'quality_score' in df.columns:
            print(f"   Quality score distribution:")
            quality_dist = df['quality_score'].value_counts().sort_index()
            for score, count in quality_dist.items():
                print(f"     Score {score}: {count:,} games")
    
    # Time control distribution
    if 'time_class' in df.columns:
        print(f"\n⏱️  Time Control Distribution:")
        time_dist = df['time_class'].value_counts()
        for tc, count in time_dist.items():
            print(f"   {tc:10}: {count:>6,} ({count/len(df)*100:.1f}%)")
    
    # ELO distribution
    if 'avg_elo' in df.columns:
        print(f"\n📈 ELO Distribution:")
        print(f"   Average ELO: {df['avg_elo'].mean():.0f}")
        print(f"   ELO range: {df['avg_elo'].min():.0f} - {df['avg_elo'].max():.0f}")
        
        # ELO brackets
        elo_brackets = [
            (800, 1200, "Beginners"),
            (1200, 1600, "Intermediate"), 
            (1600, 2000, "Advanced"),
            (2000, 2400, "Expert"),
            (2400, 3500, "Master+")
        ]
        
        print(f"   ELO bracket distribution:")
        for min_elo, max_elo, label in elo_brackets:
            count = len(df[(df['avg_elo'] >= min_elo) & (df['avg_elo'] < max_elo)])
            print(f"     {label:12} ({min_elo}-{max_elo}): {count:>6,} ({count/len(df)*100:.1f}%)")
    
    # Move analysis
    if 'actual_move_count' in df.columns:
        print(f"\n♟️  Move Analysis:")
        print(f"   Average moves per game: {df['actual_move_count'].mean():.1f}")
        print(f"   Move range: {df['actual_move_count'].min():.0f} - {df['actual_move_count'].max():.0f}")
        print(f"   Games with engine analysis: {df['has_engine_analysis'].sum():,} ({df['has_engine_analysis'].mean()*100:.1f}%)")

def main():
    # Your April 2025 file  
    file_path = r"C:\Users\mario\OneDrive\Documents\chess\dashboard\games_data\lichess_db_standard_rated_2025-04.pgn.zst"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print("🚀 Starting Enhanced Extraction with Quality Filtering")
    print("🎯 Filtering: Only rated, non-bullet games")
    print("📊 This will extract cleaner data for ML training")
    
    # Small experiment parameters
    result = extract_enhanced_games_chunked(
        file_path=file_path,
        chunk_size=100000,    # 100K games per chunk
        max_games=1000000      # 1M games total for experiment
    )
    
    if result:
        print(f"\n✅ Experiment completed successfully!")
        print(f"📄 Check the results in: data/enhanced_extraction/")
        print(f"🎯 Dataset ready for ML training - only quality games included!")

if __name__ == "__main__":
    main()