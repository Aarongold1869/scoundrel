import argparse
import json
import os


def replay(gameplay_path="gameplay.json"):
    """Replay the best gameplay saved during evaluation"""
    if not os.path.exists(gameplay_path):
        print(f"Gameplay file not found: {gameplay_path}")
        return

    with open(gameplay_path, "r") as f:
        gameplay = json.load(f)

    print(f"\nReplaying model: {gameplay.get('model_name', 'unknown')}")
    print(f"Score: {gameplay.get('score', 'unknown')} | Episode: {gameplay.get('episode', 'unknown')}")
    print("Press Enter to step through moves. Type 'q' to quit.\n")

    for step in gameplay.get("steps", []):
        room_cards = step.get("room_cards", [])
        action_str = step.get("action_str", step.get("action"))
        
        # Extract player stats
        score = step.get("score", 0)
        hp = step.get("hp", 0)
        weapon_level = step.get("weapon_level", 0)
        weapon_max = step.get("weapon_max_monster_level", 0)
        can_avoid = step.get("can_avoid", 0)
        cards_remaining = step.get("cards_remaining", 0)

        print(f"Player State: HP={hp}, Weapon={weapon_level} ({weapon_max}), Score={score}")
        print(f"Dungeon: {cards_remaining} cards remaining | Can Avoid={bool(can_avoid)}")
        
        if room_cards:
            print(f"\nCurrent Room ({len(room_cards)} cards):")
            for idx, card in enumerate(room_cards, 1):
                suit = card.get("suit", {})
                suit_symbol = suit.get("symbol", "?")
                card_symbol = card.get("symbol", "?")
                card_val = card.get("val", "?")
                card_class = suit.get("class", "?")
                print(f"  [{idx}] {card_symbol}{suit_symbol} (val: {card_val}, class: {card_class})")
        else:
            print("Current Room: (no data)")

        print(f"\n{'='*60}")
        print(f"Step {step.get('step')}: Action -> {action_str}")
        print(f"{'='*60}")

        user_input = input("\nNext step (Enter) or 'q' to quit: ")
        if user_input.strip().lower() == "q":
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay Scoundrel RL agent")

    parser.add_argument("--gameplay-path", type=str, default=None,
                        help="Path to gameplay.json for saving or replaying")
    
    args = parser.parse_args()
    replay(gameplay_path=args.gameplay_path or "gameplay.json")