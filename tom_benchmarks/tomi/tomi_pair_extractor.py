#!/usr/bin/env python3
"""
ToMi Dataset Extractor for Function Vector Research (v2)

Separates examples into tom_required vs no_tom_required groups
for contrastive activation extraction.

Usage:
    python tomi_extractor_v2.py --data_dir /path/to/tomi_balanced_story_types --output_dir /path/to/output
"""

import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class ToMiExample:
    """A single example from the ToMi dataset."""
    story: str
    question: str
    answer: str
    question_type: str
    story_type: str
    requires_tom: bool
    tom_order: Optional[int]
    base_question_type: str  # e.g., "first_order_0" without _tom/_no_tom suffix


def parse_trace_line(trace_line: str) -> Tuple[str, str, str]:
    """Parse trace line to get story_structure, question_type, story_type."""
    parts = trace_line.strip().split(',')
    story_type = parts[-1]
    question_type = parts[-2]
    story_structure = ','.join(parts[:-2])
    return story_structure, question_type, story_type


def parse_question_type(question_type: str) -> Tuple[bool, Optional[int], str]:
    """
    Parse question type.
    Returns: (requires_tom, tom_order, base_type)
    """
    if question_type in ('memory', 'reality'):
        return False, None, question_type
    
    requires_tom = question_type.endswith('_tom') and not question_type.endswith('_no_tom')
    
    if 'first_order' in question_type:
        tom_order = 1
        if '_0_' in question_type:
            base_type = 'first_order_0'
        else:
            base_type = 'first_order_1'
    elif 'second_order' in question_type:
        tom_order = 2
        if '_0_' in question_type:
            base_type = 'second_order_0'
        else:
            base_type = 'second_order_1'
    else:
        tom_order = None
        base_type = question_type
    
    return requires_tom, tom_order, base_type


def parse_story_block(lines: List[str]) -> Tuple[str, str, str]:
    """Parse a story block from the txt file."""
    story_lines = []
    question = ""
    answer = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if '\t' in line:
            parts = line.split('\t')
            q_parts = parts[0].split(' ', 1)
            question = q_parts[1] if len(q_parts) > 1 else parts[0]
            answer = parts[1]
        else:
            story_lines.append(line)
    
    story = '\n'.join(story_lines)
    return story, question, answer


def load_tomi_data(data_dir: Path, split: str = 'test') -> List[ToMiExample]:
    """Load ToMi data from txt and trace files."""
    
    # Try different filename patterns
    patterns = [
        (f'fb_all_{split}.txt', f'fb_all_{split}.trace'),
        (f'{split}.txt', f'{split}.trace'),
    ]
    
    txt_file = None
    trace_file = None
    
    for txt_pattern, trace_pattern in patterns:
        candidate_txt = data_dir / txt_pattern
        candidate_trace = data_dir / trace_pattern
        if candidate_txt.exists() and candidate_trace.exists():
            txt_file = candidate_txt
            trace_file = candidate_trace
            break
    
    if txt_file is None:
        raise FileNotFoundError(f"Could not find data files in {data_dir}")
    
    print(f"Using files: {txt_file.name}, {trace_file.name}")
    
    # Read trace file
    with open(trace_file, 'r') as f:
        trace_lines = [line.strip() for line in f if line.strip()]
    
    # Read txt file
    with open(txt_file, 'r') as f:
        txt_content = f.read()
    
    # Split into story blocks
    examples_raw = []
    current_block = []
    
    for line in txt_content.split('\n'):
        if line.startswith('1 ') and current_block:
            examples_raw.append(current_block)
            current_block = [line]
        elif line.strip():
            current_block.append(line)
    if current_block:
        examples_raw.append(current_block)
    
    print(f"Found {len(examples_raw)} story blocks and {len(trace_lines)} trace lines")
    
    # Parse examples
    tomi_examples = []
    for i, (block, trace_line) in enumerate(zip(examples_raw, trace_lines)):
        _, question_type, story_type = parse_trace_line(trace_line)
        requires_tom, tom_order, base_type = parse_question_type(question_type)
        story, question, answer = parse_story_block(block)
        
        example = ToMiExample(
            story=story,
            question=question,
            answer=answer,
            question_type=question_type,
            story_type=story_type,
            requires_tom=requires_tom,
            tom_order=tom_order,
            base_question_type=base_type
        )
        tomi_examples.append(example)
    
    return tomi_examples


def group_examples(examples: List[ToMiExample]) -> Dict[str, Dict[str, List[ToMiExample]]]:
    """
    Group examples by base question type and whether ToM is required.
    
    Returns: {base_type: {'tom': [...], 'no_tom': [...]}}
    """
    grouped = defaultdict(lambda: {'tom': [], 'no_tom': []})
    
    for ex in examples:
        # Skip control questions
        if ex.base_question_type in ('memory', 'reality'):
            continue
        
        key = 'tom' if ex.requires_tom else 'no_tom'
        grouped[ex.base_question_type][key].append(ex)
    
    return grouped


def save_grouped_data(grouped: Dict, output_dir: Path):
    """Save grouped examples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary
    summary = {}
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for base_type in sorted(grouped.keys()):
        tom_count = len(grouped[base_type]['tom'])
        no_tom_count = len(grouped[base_type]['no_tom'])
        print(f"{base_type}:")
        print(f"  ToM required:     {tom_count} examples")
        print(f"  No ToM required:  {no_tom_count} examples")
        summary[base_type] = {'tom': tom_count, 'no_tom': no_tom_count}
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save each group
    for base_type in grouped:
        for condition in ['tom', 'no_tom']:
            examples = grouped[base_type][condition]
            if not examples:
                continue
            
            filename = f'{base_type}_{condition}.jsonl'
            filepath = output_dir / filename
            
            with open(filepath, 'w') as f:
                for ex in examples:
                    ex_dict = {
                        'story': ex.story,
                        'question': ex.question,
                        'answer': ex.answer,
                        'question_type': ex.question_type,
                        'story_type': ex.story_type,
                        'requires_tom': ex.requires_tom,
                        'tom_order': ex.tom_order,
                    }
                    f.write(json.dumps(ex_dict) + '\n')
            
            print(f"Saved: {filepath}")
    
    # Save combined files for easy loading
    all_tom = []
    all_no_tom = []
    
    for base_type in grouped:
        all_tom.extend(grouped[base_type]['tom'])
        all_no_tom.extend(grouped[base_type]['no_tom'])
    
    for name, examples in [('all_tom', all_tom), ('all_no_tom', all_no_tom)]:
        filepath = output_dir / f'{name}.jsonl'
        with open(filepath, 'w') as f:
            for ex in examples:
                ex_dict = {
                    'story': ex.story,
                    'question': ex.question,
                    'answer': ex.answer,
                    'question_type': ex.question_type,
                    'story_type': ex.story_type,
                    'requires_tom': ex.requires_tom,
                    'tom_order': ex.tom_order,
                    'base_question_type': ex.base_question_type,
                }
                f.write(json.dumps(ex_dict) + '\n')
        print(f"Saved: {filepath}")
    
    # Save human-readable samples
    sample_file = output_dir / 'samples.txt'
    with open(sample_file, 'w') as f:
        for base_type in sorted(grouped.keys()):
            f.write(f"\n{'='*60}\n")
            f.write(f"QUESTION TYPE: {base_type}\n")
            f.write(f"{'='*60}\n")
            
            for condition in ['tom', 'no_tom']:
                examples = grouped[base_type][condition]
                if examples:
                    ex = examples[0]
                    f.write(f"\n--- {condition.upper()} ---\n")
                    f.write(f"Story type: {ex.story_type}\n")
                    f.write(f"Question type: {ex.question_type}\n\n")
                    f.write(ex.story + '\n\n')
                    f.write(f"Q: {ex.question}\n")
                    f.write(f"A: {ex.answer}\n")
    
    print(f"Saved samples: {sample_file}")


def main():
    parser = argparse.ArgumentParser(description='Extract ToMi examples grouped by ToM requirement')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./tomi_grouped')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    print(f"Loading ToMi {args.split} data from {data_dir}...")
    examples = load_tomi_data(data_dir, args.split)
    print(f"Loaded {len(examples)} total examples")
    
    output_dir = Path(args.output_dir)
    
    # Clear previous outputs
    if output_dir.exists():
        print(f"Clearing previous outputs in {output_dir}...")
        shutil.rmtree(output_dir)

    print("\nGrouping examples...")
    grouped = group_examples(examples)
    
    print("\nSaving...")
    save_grouped_data(grouped, output_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()