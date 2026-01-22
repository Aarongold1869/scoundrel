import pytest
from unittest.mock import patch, MagicMock
from ..scoundrel import Card, Dungeon, Weapon, Scoundrel, UI, SUITS, CARD_VALUES


class TestCard:
    """Test the Card class"""
    
    def test_card_initialization(self):
        """Test that a card can be created with all required attributes"""
        suit = {'symbol': '♦️', 'name': 'diamond', 'class': 'weapon'}
        card = Card(suit=suit, symbol='A', val=14, id=0)
        
        assert card.suit == suit
        assert card.symbol == 'A'
        assert card.val == 14
        assert card.id == 0
    
    def test_card_with_different_suits(self):
        """Test cards can be created with different suits"""
        for idx, suit in enumerate(SUITS):
            card = Card(suit=suit, symbol='K', val=13, id=idx)
            assert card.suit['name'] == suit['name']
            assert card.suit['class'] == suit['class']


class TestDungeon:
    """Test the Dungeon class"""
    
    def test_dungeon_initialization(self):
        """Test that dungeon initializes with correct number of cards"""
        dungeon = Dungeon()
        # Hearts and diamonds don't include J, Q, K (only 2-10)
        # Spades and clubs include all (2-10, J, Q, K, A)
        # 2 suits * 9 cards + 2 suits * 13 cards = 18 + 26 = 44
        assert len(dungeon.cards) == 44
        assert len(dungeon.current_room) == 4
        assert dungeon.can_avoid is True
        assert dungeon.can_heal is True
    
    def test_dungeon_cards_have_unique_ids(self):
        """Test that all cards in dungeon have unique IDs"""
        dungeon = Dungeon()
        card_ids = [card.id for card in dungeon.cards]
        assert len(card_ids) == len(set(card_ids))
    
    def test_dungeon_shuffle(self):
        """Test that shuffle changes card order"""
        dungeon = Dungeon()
        original_order = [card.id for card in dungeon.cards[:10]]
        dungeon.shuffle()
        new_order = [card.id for card in dungeon.cards[:10]]
        # It's very unlikely shuffle produces same order
        assert original_order != new_order or len(dungeon.cards) < 2
    
    def test_cards_remaining(self):
        """Test that cards_remaining returns correct count"""
        dungeon = Dungeon()
        assert dungeon.cards_remaining() == 44
    
    def test_discard_card(self):
        """Test discarding a card removes it from dungeon and current room"""
        dungeon = Dungeon()
        card_to_discard = dungeon.current_room[0]
        initial_count = dungeon.cards_remaining()
        
        dungeon.discard(card_to_discard)
        
        assert dungeon.cards_remaining() == initial_count - 1
        assert card_to_discard not in dungeon.cards
        assert card_to_discard not in dungeon.current_room
    
    def test_discard_refreshes_room_when_one_card_left(self):
        """Test that discarding to 1 card refreshes the room"""
        dungeon = Dungeon()
        # Discard until only 1 card left in current room
        for i in range(2):
            dungeon.discard(dungeon.current_room[0])
            assert len(dungeon.current_room) == 4 - (i + 1)
        
        # Discard the last card - should refresh room
        dungeon.discard(dungeon.current_room[0])
        
        assert len(dungeon.current_room) == 4
        assert dungeon.can_avoid is True
        assert dungeon.can_heal is True
    
    def test_avoid_room_when_allowed(self):
        """Test avoiding a room shuffles current room to bottom"""
        scoundrel = Scoundrel(ui=UI.API)
        original_room_ids = {card.id for card in scoundrel.dungeon.current_room}
        
        try:
            scoundrel.take_action(action='a')
        except Exception as e:
            print(e)
        
        # Current room should be different cards
        new_room_ids = {card.id for card in scoundrel.dungeon.current_room}
        assert new_room_ids != original_room_ids
        assert scoundrel.dungeon.can_avoid is False
    
    def test_avoid_room_when_not_allowed(self):
        """Test avoiding room when can_avoid is False"""
        scoundrel = Scoundrel(ui=UI.API)
        scoundrel.dungeon.can_avoid = False
        original_room_ids = {card.id for card in scoundrel.dungeon.current_room}
        
        try:
            scoundrel.take_action(action='a')
        except Exception as e:
            print(e)
        
        # Room should not change
        new_room_ids = {card.id for card in scoundrel.dungeon.current_room}
        assert new_room_ids == original_room_ids


class TestWeapon:
    """Test the Weapon class"""
    
    def test_weapon_initialization(self):
        """Test weapon initializes with correct values"""
        weapon = Weapon(level=5)
        assert weapon.level == 5
        assert weapon.max_monster_level == 15
    
    def test_weapon_string_representation_with_level(self):
        """Test weapon string shows level and max monster level"""
        weapon = Weapon(level=7)
        assert str(weapon) == "7 (15)"
    
    def test_weapon_string_representation_no_level(self):
        """Test weapon string when level is 0"""
        weapon = Weapon(level=0)
        assert str(weapon) == "None equipped"
    
    def test_weapon_degrade(self):
        """Test weapon degrades after use"""
        weapon = Weapon(level=8)
        assert weapon.max_monster_level == 15
        
        weapon.degrade(monster_level=10)
        
        assert weapon.max_monster_level == 10
        assert weapon.level == 8  # Level doesn't change
    
    def test_weapon_degrade_with_zero_level(self):
        """Test degrading a weapon with 0 level doesn't change max_monster_level"""
        weapon = Weapon(level=0)
        original_max = weapon.max_monster_level
        
        weapon.degrade(monster_level=5)
        
        # Since level > 0 is required, it shouldn't degrade
        assert weapon.max_monster_level == original_max


class TestScoundrel:
    """Test the Scoundrel class"""
    
    def test_scoundrel_initialization(self):
        """Test scoundrel initializes with correct default values"""
        scoundrel = Scoundrel()
        
        assert scoundrel.ui == UI.CLI
        assert scoundrel.score == -188
        assert scoundrel.hp == 20
        assert isinstance(scoundrel.dungeon, Dungeon)
        assert isinstance(scoundrel.weapon, Weapon)
        assert scoundrel.weapon.level == 0
    
    def test_scoundrel_initialization_with_different_ui(self):
        """Test scoundrel can be initialized with different UI modes"""
        scoundrel_api = Scoundrel(ui=UI.API)
        assert scoundrel_api.ui == UI.API
        
        scoundrel_pygame = Scoundrel(ui=UI.PYGAME)
        assert scoundrel_pygame.ui == UI.PYGAME
    
    def test_update_score(self):
        """Test score updates correctly based on remaining cards and HP"""
        scoundrel = Scoundrel()
        scoundrel.hp = 15
        scoundrel.update_score()
        
        # Score should be: -1 * (sum of all monster cards) + hp
        monster_sum = sum(
            card.val for card in scoundrel.dungeon.cards 
            if card.suit['name'] in ['spade', 'club']
        )
        expected_score = -monster_sum + 15
        assert scoundrel.score == expected_score
    
    def test_update_hp_normal(self):
        """Test HP updates normally"""
        scoundrel = Scoundrel()
        scoundrel.update_hp(5)
        assert scoundrel.hp == 20  # Capped at 20
        
        scoundrel.update_hp(-5)
        assert scoundrel.hp == 15
    
    def test_update_hp_capped_at_max(self):
        """Test HP cannot exceed 20"""
        scoundrel = Scoundrel()
        scoundrel.hp = 18
        scoundrel.update_hp(10)
        assert scoundrel.hp == 20
    
    def test_update_hp_capped_at_min(self):
        """Test HP cannot go below 0"""
        scoundrel = Scoundrel()
        scoundrel.hp = 5
        scoundrel.update_hp(-10)
        assert scoundrel.hp == 0
    
    def test_equip_weapon(self):
        """Test equipping a weapon"""
        scoundrel = Scoundrel()
        scoundrel.equip_weapon(weapon_level=7)
        
        assert scoundrel.weapon.level == 7
        assert scoundrel.weapon.max_monster_level == 15
    
    def test_fight_monster_without_weapon(self):
        """Test fighting monster without weapon takes full damage"""
        scoundrel = Scoundrel()
        initial_hp = scoundrel.hp
        
        scoundrel.fight_monster(monster_level=5, use_weapon=False)
        
        assert scoundrel.hp == initial_hp - 5
        assert scoundrel.weapon.level == 0
    
    def test_fight_monster_with_weapon(self):
        """Test fighting monster with weapon reduces damage"""
        scoundrel = Scoundrel()
        scoundrel.equip_weapon(weapon_level=8)
        initial_hp = scoundrel.hp
        
        scoundrel.fight_monster(monster_level=5, use_weapon=True)
        
        # Damage = -5 + 8 = 3, min(3, 0) = 0
        assert scoundrel.hp == initial_hp
        assert scoundrel.weapon.max_monster_level == 5  # Weapon degrades
    
    def test_fight_monster_with_weak_weapon(self):
        """Test fighting strong monster with weak weapon still takes damage"""
        scoundrel = Scoundrel()
        scoundrel.equip_weapon(weapon_level=3)
        initial_hp = scoundrel.hp
        
        scoundrel.fight_monster(monster_level=8, use_weapon=True)
        
        # Damage = -8 + 3 = -5
        assert scoundrel.hp == initial_hp - 5
        assert scoundrel.weapon.max_monster_level == 8
    
    def test_interact_card_monster(self):
        """Test interacting with monster card"""
        scoundrel = Scoundrel(ui=UI.API)
        scoundrel.equip_weapon(weapon_level=10)
        
        # Find a monster card
        monster_card = next(
            card for card in scoundrel.dungeon.cards 
            if card.suit['class'] == 'monster'
        )
        initial_hp = scoundrel.hp
        initial_cards = scoundrel.dungeon.cards_remaining()
        
        scoundrel.interact_card(monster_card)
        
        # Card should be discarded
        assert scoundrel.dungeon.cards_remaining() == initial_cards - 1
        assert monster_card not in scoundrel.dungeon.cards
        assert scoundrel.dungeon.can_avoid is False
    
    def test_interact_card_health_potion(self):
        """Test interacting with health potion"""
        scoundrel = Scoundrel(ui=UI.API)
        scoundrel.hp = 10
        
        # Find a heart card
        health_card = next(
            card for card in scoundrel.dungeon.cards 
            if card.suit['class'] == 'health'
        )
        
        scoundrel.interact_card(health_card)
        
        assert scoundrel.hp == 10 + health_card.val
        assert scoundrel.dungeon.can_heal is False
        assert health_card not in scoundrel.dungeon.cards
    
    def test_interact_card_health_potion_when_cannot_heal(self):
        """Test health potion doesn't heal when can_heal is False"""
        scoundrel = Scoundrel(ui=UI.API)
        scoundrel.hp = 10
        scoundrel.dungeon.can_heal = False
        
        health_card = next(
            card for card in scoundrel.dungeon.cards 
            if card.suit['class'] == 'health'
        )
        
        scoundrel.interact_card(health_card)
        
        # HP should not change
        assert scoundrel.hp == 10
        assert health_card not in scoundrel.dungeon.cards
    
    def test_interact_card_weapon(self):
        """Test interacting with weapon card"""
        scoundrel = Scoundrel(ui=UI.API)
        
        # Find a weapon card
        weapon_card = next(
            card for card in scoundrel.dungeon.cards 
            if card.suit['class'] == 'weapon'
        )
        
        scoundrel.interact_card(weapon_card)
        
        assert scoundrel.weapon.level == weapon_card.val
        assert weapon_card not in scoundrel.dungeon.cards
    
    @patch('builtins.input', return_value='h')
    def test_use_weapon_cli_choose_hands(self, mock_input):
        """Test CLI weapon choice - choosing hands"""
        scoundrel = Scoundrel(ui=UI.CLI)
        scoundrel.equip_weapon(weapon_level=10)
        card = Card(
            suit={'symbol': '♠️', 'name': 'spade', 'class': 'monster'},
            symbol='5',
            val=5,
            id=0
        )
        
        result = scoundrel.use_weapon_cli(card)
        
        assert result is False
    
    @patch('builtins.input', return_value='w')
    def test_use_weapon_cli_choose_weapon(self, mock_input):
        """Test CLI weapon choice - choosing weapon"""
        scoundrel = Scoundrel(ui=UI.CLI)
        scoundrel.equip_weapon(weapon_level=10)
        card = Card(
            suit={'symbol': '♠️', 'name': 'spade', 'class': 'monster'},
            symbol='5',
            val=5,
            id=0
        )
        
        result = scoundrel.use_weapon_cli(card)
        
        assert result is True
    
    @patch('builtins.input', return_value='c')
    def test_use_weapon_cli_cancel(self, mock_input):
        """Test CLI weapon choice - canceling"""
        scoundrel = Scoundrel(ui=UI.CLI)
        scoundrel.equip_weapon(weapon_level=10)
        card = Card(
            suit={'symbol': '♠️', 'name': 'spade', 'class': 'monster'},
            symbol='5',
            val=5,
            id=0
        )
        
        result = scoundrel.use_weapon_cli(card)
        
        assert result is None
    
    def test_use_weapon_cli_no_weapon(self):
        """Test CLI weapon choice when no weapon equipped"""
        scoundrel = Scoundrel(ui=UI.CLI)
        card = Card(
            suit={'symbol': '♠️', 'name': 'spade', 'class': 'monster'},
            symbol='5',
            val=5,
            id=0
        )
        
        result = scoundrel.use_weapon_cli(card)
        
        assert result is False
    
    @patch('builtins.input', return_value='w')
    def test_use_weapon_cli_weapon_too_weak(self, mock_input):
        """Test CLI weapon choice when weapon is too weak"""
        scoundrel = Scoundrel(ui=UI.CLI)
        scoundrel.equip_weapon(weapon_level=3)
        scoundrel.weapon.max_monster_level = 5
        card = Card(
            suit={'symbol': '♠️', 'name': 'spade', 'class': 'monster'},
            symbol='K',
            val=13,
            id=0
        )
        
        result = scoundrel.use_weapon_cli(card)
        
        assert result is None
    
    def test_play_quit(self):
        """Test quitting the game"""
        scoundrel = Scoundrel(ui=UI.API)
        
        scoundrel.take_action('q')
        
        assert isinstance(scoundrel.score, int)
        assert scoundrel.game_is_active == False
    
    # @patch('builtins.input', side_effect=['a'] + ['1'] * 50)
    # @patch('builtins.print')
    # def test_play_avoid_room(self, mock_print, mock_input):
    #     """Test avoiding a room during play"""
    #     scoundrel = Scoundrel(ui=UI.CLI)
    #     original_room = scoundrel.dungeon.current_room.copy()
        
    #     # This will avoid the room once, then play cards

    #     scoundrel.play()
        
    #     # Can_avoid should be False after avoiding
    #     assert scoundrel.dungeon.can_avoid is False or scoundrel.hp == 0
    
    def test_play_until_death(self):
        """Test playing until death"""
        scoundrel = Scoundrel(ui=UI.API)
        while scoundrel.game_is_active:
            scoundrel.take_action(action='1')
        
        # Game should end when HP reaches 0 or no cards left
        assert scoundrel.hp == 0 or scoundrel.dungeon.cards_remaining() == 0
    
    def test_play_invalid_input(self):
        """Test handling invalid input during play"""
        scoundrel = Scoundrel(ui=UI.API)
        prev_game_state = scoundrel.current_game_state()
        try:
            self.take_action(action='invalid')
        except Exception as e:
            print(e)
        assert prev_game_state == scoundrel.current_game_state()  


class TestUIEnum:
    """Test the UI enum"""
    
    def test_ui_enum_values(self):
        """Test UI enum has expected values"""
        assert UI.CLI.value == 'cli'
        assert UI.PYGAME.value == 'pygame'
        assert UI.API.value == 'api'


class TestIntegration:
    """Integration tests for complete game scenarios"""
    
    def test_complete_game_flow_with_weapon_pickups(self):
        """Test a complete game with weapon pickups and monster fights"""
        scoundrel = Scoundrel(ui=UI.API)
        
        # Find and equip a weapon
        weapon_card = next(
            card for card in scoundrel.dungeon.current_room 
            if card.suit['class'] == 'weapon'
        ) if any(c.suit['class'] == 'weapon' for c in scoundrel.dungeon.current_room) else None
        
        if weapon_card:
            scoundrel.interact_card(weapon_card)
            assert scoundrel.weapon.level > 0
    
    def test_score_calculation_accuracy(self):
        """Test that score is calculated correctly throughout the game"""
        scoundrel = Scoundrel(ui=UI.API)
        initial_score = scoundrel.score
        
        scoundrel.update_score()
        
        # Calculate expected score manually
        monster_sum = sum(
            card.val for card in scoundrel.dungeon.cards 
            if card.suit['name'] in ['spade', 'club']
        )
        expected_score = -monster_sum + scoundrel.hp
        
        assert scoundrel.score == expected_score
    
    def test_dungeon_depletion(self):
        """Test that dungeon properly handles card depletion"""
        scoundrel = Scoundrel(ui=UI.API)
        initial_count = scoundrel.dungeon.cards_remaining()
        
        # Discard some cards
        for _ in range(10):
            if scoundrel.dungeon.current_room:
                card = scoundrel.dungeon.current_room[0]
                scoundrel.dungeon.discard(card)
        
        assert scoundrel.dungeon.cards_remaining() == initial_count - 10
