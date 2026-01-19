import random
from typing import List


SUITS = [
    {'symbol':'♦️', 'name': 'diamond', 'class': 'weapon'},
    {'symbol':'♥️', 'name': 'heart', 'class': 'health'},
    {'symbol':'♠️', 'name': 'spade', 'class': 'monster'},
    {'symbol':'♣️', 'name': 'club', 'class': 'monster'}
]


CARD_VALUES = {
    **{str(i): i for i in range(2, 11)},
    'J': 11,
    'Q': 12,
    'K': 13,
    'A': 14
}


class Card():
    
    def __init__(
            self,
            suit: str,
            symbol: str,
            val: int,
            id: int
            ):
        self.suit = suit
        self.symbol = symbol
        self.val = val
        self.id = id


class Dungeon():
    
    def __init__(self):
        self.cards: List[Card] = []
        i = 0
        for suit in SUITS:
            for symbol, val in CARD_VALUES.items():
                if suit['name'] in ['heart', 'diamond'] and val > 10:
                    continue
                self.cards.append(Card(suit=suit, symbol=symbol, val=val, id=i))
                i += 1

        self.shuffle()
        self.current_room = self.cards[:4]
        self.can_avoid = True
        self.can_heal = True
        
    def shuffle(self) -> List[Card]:
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

    def avoid_room(self):
        if not self.can_avoid:
            print('cannot avoid.')
            return
        random.shuffle(self.current_room)
        self.cards = self.cards[4:] + self.current_room
        self.current_room = self.cards[:4]
        self.can_avoid = False
        


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


class Scoundrel():

    def __init__(self):
        self.score = -188
        self.hp = 20
        self.dungeon = Dungeon()
        self.weapon = None
        
    def update_score(self):
        self.score = sum(card.val * -1 for card in self.dungeon.cards if card.suit['name'] not in ['heart', 'diamond']) + self.hp

    def update_hp(self, val: int):
        self.hp += val
        if self.hp > 20:
            self.hp = 20
        if self.hp < 0:
            self.hp = 0

    def equip_weapon(self, weapon_level: int):
        self.weapon = Weapon(level=weapon_level)

    def fight_monster(self, monster_level: int, use_weapon: bool):
        damage = monster_level * -1
        if use_weapon:
            damage = min(damage + self.weapon.level, 0)
            self.weapon.degrade(monster_level=monster_level)
        self.update_hp(val=damage)
        
    def interact_card(self, card: Card):
        if card.suit['class'] == 'monster':
            # print(f'Fight Monster: {card.val}')
            fight_with = input("\nHow fight?\nUse hands (h). Use weapon (w).")

            if fight_with == 'h':
                self.fight_monster(monster_level=card.val,
                                   use_weapon=False)
            
            elif fight_with == 'w':
                if self.weapon and card.val < self.weapon.max_monster_level:
                    self.fight_monster(monster_level=card.val,
                                       use_weapon=True)
                elif not self.weapon:
                    print('no weapon equipped.')
                    return
                
                elif card.val >= self.weapon.max_monster_level:
                    print('weapon is not strong enough.')
                    return
                     
            else:
                print('invalid action.')
                return
                    
        elif card.suit['class'] == 'health':
            # print(f'Health potion: {card.val}')
            if self.dungeon.can_heal:
                self.update_hp(val=card.val)
            self.dungeon.can_heal = False

        elif card.suit['class'] == 'weapon':
            # print(f'Equip weapon: {card.val}')
            self.equip_weapon(weapon_level=card.val)

        self.dungeon.can_avoid = False
        self.dungeon.discard(discard=card)
        self.update_score()

    def play(self):
        while self.hp > 0 and self.dungeon.cards_remaining() > 0:
            print(f'\nHP: {self.hp} | Score: {self.score} | Weapon: {self.weapon} | Cards remaining: {self.dungeon.cards_remaining()} \n')
            # print('Score: ', self.score)
            # print('Weapon: ', self.weapon)
            # print('Cards remaining: ', self.dungeon.cards_remaining())
            self.dungeon.display_current_room()
            # self.dungeon.display()
            action = input("\nwhat do? ")
            
            if action in ['1', '2', '3', '4']:
                try:
                    card_index = int(action) -1
                    self.interact_card(card=self.dungeon.current_room[card_index])
                except IndexError:
                    print('invalid action.')

            elif action == "a":
                self.dungeon.avoid_room()

            elif action == 'q':
                return self.score

            else:
                print('invalid action.')
        
        if self.score > 0:
            print('\nYou win!')
        else:
            print('\nYou lose.')

        print('Score: ', self.score)
        return self.score
