from enum import Enum
import random
from typing import Dict, List, Literal, Optional, TypedDict


SUITS = [
    {'symbol':'♦️', 'name': 'diamond', 'class': 'weapon'},
    {'symbol':'♥️', 'name': 'heart', 'class': 'health'},
    {'symbol':'♠️', 'name': 'spade', 'class': 'monster'},
    {'symbol':'♣️', 'name': 'club', 'class': 'monster'}
]


CARD_VALUES = {
    'A': 14,
    'K': 13,
    'Q': 12,
    'J': 11,
    **{str(i): i for i in range(10, 1, -1)}
}


class Card():
    
    def __init__(
            self,
            suit: Dict[str, str],
            symbol: str,
            val: int,
            id: int
            ):
        self.suit = suit
        self.symbol = symbol
        self.val = val
        self.id = id

    def __str__(self):
        return f"{self.symbol}{self.suit['symbol']}"

    def __repr__(self):
        return self.__str__()


class DungeonState(TypedDict):
    cards_remaining: int
    monsters_remaining: int
    monster_strength_remaining: int
    weapons_remaining: int
    weapon_strength_remaining: int
    potions_remaining: int
    potion_strength_remaining: int

class RoomState(TypedDict):
    cards_remaining: int
    cards: List[Card]
    can_avoid: int
    can_heal: int

class Dungeon():
    
    def __init__(self, shuffle_deck: bool = True):
        self.cards: List[Card] = []
        i = 0
        for suit in SUITS:
            for symbol, val in CARD_VALUES.items():
                if suit['name'] in ['heart', 'diamond'] and val > 10:
                    continue
                self.cards.append(Card(suit=suit, symbol=symbol, val=val, id=i))
                i += 1

        if shuffle_deck:
            self.shuffle()
        self.monsters_remaining = 26
        self.monster_strength_remaining = 208
        self.weapons_remaining = 9
        self.weapon_strength_remaining = 54
        self.potions_remaining = 9
        self.potion_strength_remaining = 54
        self.current_room = self.cards[:4]
        self.can_avoid = True
        self.can_heal = True
        
    def shuffle(self) -> None:
        random.shuffle(self.cards)

    def display(self):
        print([f"{card.symbol}{card.suit['symbol']}" for card in self.cards])

    def display_current_room(self):
        print([f"{card.symbol}{card.suit['symbol']}" for card in self.current_room])

    def cards_remaining(self):
        return len(self.cards)

    def discard(self, discard: Card):

        self.cards = list(filter(lambda card: card.id != discard.id, self.cards))
        self.current_room = list(filter(lambda card: card.id != discard.id, self.current_room))
        if len(self.current_room) == 1 and self.cards_remaining() > 1:
            self.current_room = self.cards[:4]
            self.can_avoid = True
            self.can_heal = True

    def avoid_room(self) -> None:
        if not self.can_avoid:
            raise ValueError('cannot avoid.')
        random.shuffle(self.current_room)
        self.cards = self.cards[len(self.current_room):] + self.current_room
        self.current_room = self.cards[:4]
        self.can_avoid = False

    def room_state(self) -> RoomState:
        return RoomState(
            cards_remaining=len(self.current_room),
            cards=self.current_room,
            can_avoid=1 if self.can_avoid else 0,
            can_heal=1 if self.can_heal else 0,
        )
    
    def dungeon_state(self) -> DungeonState:
        return DungeonState(
            cards_remaining=self.cards_remaining(),
            monsters_remaining=self.monsters_remaining,
            monster_strength_remaining=self.monster_strength_remaining,
            weapons_remaining=self.weapons_remaining,
            weapon_strength_remaining=self.weapon_strength_remaining,
            potions_remaining=self.potions_remaining,
            potion_strength_remaining=self.potion_strength_remaining
        )

        
class Weapon():

    def __init__(self, level: int):
        self.level = level
        self.max_monster_level = 15 # can hit Ace

    def __str__(self):
        if self.level == 0:
            return "None equipped"
        return f"{self.level} ({self.max_monster_level})"
    
    def degrade(self, monster_level):
        if self.level > 0:
            self.max_monster_level = monster_level


class UI(Enum):
    CLI = 'cli'
    PYGAME = 'pygame'
    API = 'api'


class PlayerState(TypedDict):
    hp: int
    weapon_level: int
    weapon_max_monster_level: int


class GameState(TypedDict):
    # game data 
    is_active: bool
    score: int
    # player data 
    player_state: PlayerState
    # dungeon data
    dungeon_state: DungeonState
    # room data
    room_state: RoomState


class Scoundrel():

    def __init__(self, ui: UI = UI.CLI):
        self.ui = ui
        self.dungeon = Dungeon()
        self.hp = 20
        self.score = (self.dungeon.monster_strength_remaining * -1) + self.hp
        self.weapon = Weapon(level=0)
        self.game_is_active = True

    def update_score(self):
        self.score = (self.dungeon.monster_strength_remaining * -1) + self.hp

    def update_hp(self, val: int):
        self.hp += val
        if self.hp > 20:
            self.hp = 20
        if self.hp < 0:
            self.hp = 0

    def equip_weapon(self, weapon_level: int):
        self.weapon = Weapon(level=weapon_level)

    def fight_monster(self, monster_level: int, use_weapon: bool):
        # print(f'use weapon: {use_weapon}')
        damage = monster_level * -1
        if use_weapon:
            damage = min(damage + self.weapon.level, 0)
            self.weapon.degrade(monster_level=monster_level)
        self.update_hp(val=damage)

    def use_weapon_cli(self, card: Card) -> Optional[bool]:

        if self.weapon.level == 0:
            return False

        fight_with = input("\nHow fight?\nUse hands (h). Use weapon (w). Cancel (c).")

        if fight_with == 'c':
            return None

        if fight_with == 'h':
            return False
        
        elif fight_with == 'w':
            if self.weapon and card.val < self.weapon.max_monster_level:
                return True
            
            elif not self.weapon:
                print('no weapon equipped.')
                return False
            
            elif card.val >= self.weapon.max_monster_level:
                print('weapon is not strong enough.')
                return None
                
        else:
            print('invalid action.')
            return None
        
        return None
    
    def interact_card(self, card: Card):
        if card.suit['class'] == 'monster':

            if self.ui == UI.CLI:
                use_weapon = None
                while use_weapon is None:
                    use_weapon = self.use_weapon_cli(card=card)
            
            else: 
                use_weapon = card.val < self.weapon.max_monster_level

            self.fight_monster(monster_level=card.val,
                                       use_weapon=use_weapon)
            self.dungeon.monsters_remaining -= 1
            self.dungeon.monster_strength_remaining -= card.val
                    
        elif card.suit['class'] == 'health':
            # print(f'Health potion: {card.val}')
            if self.dungeon.can_heal:
                self.update_hp(val=card.val)
            self.dungeon.can_heal = False
            self.dungeon.potions_remaining -= 1
            self.dungeon.potion_strength_remaining -= card.val

        elif card.suit['class'] == 'weapon':
            # print(f'Equip weapon: {card.val}')
            self.equip_weapon(weapon_level=card.val)
            self.dungeon.weapons_remaining -= 1
            self.dungeon.weapon_strength_remaining -= card.val

        self.dungeon.can_avoid = False
        self.dungeon.discard(discard=card)
        self.update_score()
        self.game_is_active = self.hp > 0 and self.dungeon.cards_remaining() > 0

    def current_game_state(self) -> GameState:
        player_state = PlayerState(hp=self.hp,
                                   weapon_level=self.weapon.level,
                                   weapon_max_monster_level=self.weapon.max_monster_level)
        return GameState(
            is_active=self.game_is_active,
            score=self.score,
            player_state=player_state,
            dungeon_state=self.dungeon.dungeon_state(),
            room_state=self.dungeon.room_state()
        )
    
    def take_action(self, action: str) -> None:

        if action in ['1', '2', '3', '4']:
            try:
                card_index = int(action) -1
                card = self.dungeon.current_room[card_index]
                self.interact_card(card=card) 

            except IndexError:
                # print(f'no card at index {action}')
                raise ValueError(f'no card at index {action}')
            
            except Exception as e:
                # print('invalid action')
                raise ValueError(f'invalid action: {action}. {e}')

        elif action == "a":
            self.dungeon.avoid_room()

        elif action == 'q':
            self.game_is_active = False

        else:
            # print('invalid action')
            raise ValueError(f'invalid action: {action}.')

    def play(self):
        if self.ui == UI.CLI:
            while self.game_is_active:

                print(f'\nHP: {self.hp} | Score: {self.score} | Weapon: {self.weapon} | Cards remaining: {self.dungeon.cards_remaining()} \n')
                self.dungeon.display_current_room()
                # self.dungeon.display()
                action = input("\nwhat do? ")
                try:
                    self.take_action(action=action)
                except Exception as e:
                    print(e)
            
            if self.score > 0:
                print('\nYou win!')
            else:
                print('\nYou lose.')

            print('Score: ', self.score)
            return self.score
    
