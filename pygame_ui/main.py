from scoundrel.scoundrel import Scoundrel, Card

import os
from pathlib import Path
import pygame
from pygame_emojis import load_emoji
import sys

# Initialize pygame
pygame.init()

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')

# Screen settings
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Scoundrel - Card Dungeon Crawler")

background = pygame.image.load(os.path.join(ASSETS_DIR, "images/dungeon_background.png")).convert_alpha()

# Scale background to fit screen
background = pygame.transform.scale(background, (SCREEN_WIDTH, SCREEN_HEIGHT))

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (220, 53, 69)
GREEN = (40, 167, 69)
BLUE = (0, 123, 255)
YELLOW = (255, 193, 7)
GRAY = (108, 117, 125)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)

# Card settings
CARD_WIDTH = 150 * 1.2
CARD_HEIGHT = 220 * 1.2
CARD_SPACING = 30

CARD_SURFACE = pygame.Surface((CARD_WIDTH, CARD_HEIGHT), pygame.SRCALPHA)
# Load the paper patina image
CARD_BG = pygame.image.load(os.path.join(ASSETS_DIR, "images/card2.png")).convert_alpha()
# (Optional) Ensure it matches the surface size
CARD_BG = pygame.transform.scale(CARD_BG, (CARD_WIDTH, CARD_HEIGHT))
CARD_SURFACE.blit(CARD_BG, (0, 0))

# Fonts
FONT_FILE = os.path.join(ASSETS_DIR, "fonts/MedievalSharp-Regular.ttf")
# FONT_FILE = None

title_font = pygame.font.Font(FONT_FILE, 48)
card_font = pygame.font.Font(FONT_FILE, 72)
info_font = pygame.font.Font(FONT_FILE, 32)
small_font = pygame.font.Font(FONT_FILE, 24)

# Button class
class Button:
    def __init__(self, x, y, width, height, text, color, text_color=WHITE):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.hovered = False
        self.disabled = False

    def draw(self, surface):
        if self.disabled:
            # Draw disabled button in gray
            color = GRAY
            border_color = DARK_GRAY
            text_color = DARK_GRAY
        else:
            color = tuple(min(c + 30, 255) for c in self.color) if self.hovered else self.color
            border_color = WHITE
            text_color = self.text_color
            
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        pygame.draw.rect(surface, border_color, self.rect, 2, border_radius=8)
        
        text_surface = info_font.render(self.text, True, text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos) and not self.disabled

    def update_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos) and not self.disabled


# Card visual class
class CardVisual:
    def __init__(self, card: Card, x, y, index):
        self.card = card
        self.rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)
        self.index = index + 1
        self.hovered = False

    def draw(self, surface):
        # Card background based on suit class
        if self.card.suit['class'] == 'monster':
            color = BLACK
        elif self.card.suit['class'] == 'health':
            color = RED
        elif self.card.suit['class'] == 'weapon':
            color = BLUE
        else:
            color = GRAY

        # Lighten color if hovered
        if self.hovered:
            color = tuple(min(c + 40, 255) for c in color)

        # Draw card background
        screen.blit(CARD_SURFACE, self.rect)
        # pygame.draw.rect(surface, color, self.rect, border_radius=12)
        # pygame.draw.rect(surface, WHITE, self.rect, 3, border_radius=12)

        # Draw card suit
        suit_image_surface = load_emoji(self.card.suit['symbol'], 42)
        suit_image_rect = suit_image_surface.get_rect(topright=(self.rect.x + CARD_WIDTH - 20, self.rect.y + 25))
        surface.blit(suit_image_surface, suit_image_rect)

        # Draw card symbol and suit
        symbol_text = self.card.symbol
        symbol_surface = card_font.render(symbol_text, True, WHITE)
        symbol_rect = symbol_surface.get_rect(center=self.rect.center)
        surface.blit(symbol_surface, symbol_rect)

        # Draw card value
        val_text = str(self.card.val)
        val_surface = info_font.render(val_text, True, WHITE)
        val_rect = val_surface.get_rect(topleft=(self.rect.x + 25, self.rect.y + 30))
        surface.blit(val_surface, val_rect)

        # Draw index number at bottom
        index_text = f"Press {self.index}"
        index_surface = small_font.render(index_text, True, WHITE)
        index_rect = index_surface.get_rect(center=(self.rect.centerx, self.rect.bottom - 15))
        surface.blit(index_surface, index_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

    def update_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)


# Main game class
class ScoundrelGame:
    def __init__(self):
        self.scoundrel = Scoundrel()
        self.running = True
        self.game_over = False
        self.game_over_message = ""
        
        # Create avoid button
        self.avoid_button = Button(
            SCREEN_WIDTH // 2 - 140,
            SCREEN_HEIGHT - 110,
            280,
            50,
            "Avoid Room (A)",
            YELLOW,
            BLACK
        )

        # Create quit button
        self.quit_button = Button(
            SCREEN_WIDTH - 150,
            20,
            120,
            40,
            "Quit (Q)",
            GRAY
        )

    def create_card_visuals(self):
        """Create visual representations of the current room cards"""
        cards = []
        total_width = len(self.scoundrel.dungeon.current_room) * CARD_WIDTH + \
                     (len(self.scoundrel.dungeon.current_room) - 1) * CARD_SPACING
        start_x = (SCREEN_WIDTH - total_width) // 2
        card_y = 225

        for i, card in enumerate(self.scoundrel.dungeon.current_room):
            x = start_x + i * (CARD_WIDTH + CARD_SPACING)
            cards.append(CardVisual(card, x, card_y, i))
        
        return cards

    def draw_stats(self, surface):
        """Draw player stats at the top"""
        # y_pos = 50
        x_pos = 50

        # HP
        hp_text = f"HP: {self.scoundrel.hp}/20"
        hp_color = GREEN if self.scoundrel.hp > 10 else (YELLOW if self.scoundrel.hp > 5 else RED)
        hp_surface = info_font.render(hp_text, True, hp_color)
        surface.blit(hp_surface, (x_pos, 50))
        
        # Weapon
        weapon_text = f"Weapon: {self.scoundrel.weapon}"
        weapon_surface = info_font.render(weapon_text, True, WHITE)
        surface.blit(weapon_surface, (x_pos, 80))

    def draw_cards_remaining(self, surface):
        """Draw cards remaining under the avoid button"""
        cards_text = f"Cards remaining: {self.scoundrel.dungeon.cards_remaining()}"
        cards_surface = info_font.render(cards_text, True, WHITE)
        cards_rect = cards_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 140))
        surface.blit(cards_surface, cards_rect)

    def draw_score(self, surface):
        """Draw game score"""
        # Score
        score_text = f"Score: {self.scoundrel.score}"
        score_color = GREEN if self.scoundrel.score > 0 else RED
        score_surface = title_font.render(score_text, True, score_color)
        score_rect = score_surface.get_rect(center=(SCREEN_WIDTH // 2, 150))
        surface.blit(score_surface, score_rect)

    def draw_game_over(self, surface):
        """Draw game over screen"""
        # Semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(200)
        overlay.fill(BLACK)
        surface.blit(overlay, (0, 0))

        # Game over message
        game_over_surface = title_font.render(self.game_over_message, True, 
                                              GREEN if self.scoundrel.score > 0 else RED)
        game_over_rect = game_over_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
        surface.blit(game_over_surface, game_over_rect)

        # Final score
        score_text = f"Final Score: {self.scoundrel.score}"
        score_surface = info_font.render(score_text, True, WHITE)
        score_rect = score_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
        surface.blit(score_surface, score_rect)

        # Instructions
        restart_text = "Press R to Restart or Q to Quit"
        restart_surface = small_font.render(restart_text, True, LIGHT_GRAY)
        restart_rect = restart_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 80))
        surface.blit(restart_surface, restart_rect)

    def handle_card_interaction(self, card: Card):
        """Handle interaction with a card"""
        self.scoundrel.interact_card(card)
        self.check_game_over()

    def check_game_over(self):
        """Check if game is over"""
        if self.scoundrel.hp <= 0:
            self.game_over = True
            self.game_over_message = "YOU DIED!"
        elif self.scoundrel.dungeon.cards_remaining() == 0:
            self.game_over = True
            if self.scoundrel.score > 0:
                self.game_over_message = "YOU WIN!"
            else:
                self.game_over_message = "YOU LOSE!"

    def reset_game(self):
        """Reset the game"""
        self.scoundrel = Scoundrel()
        self.game_over = False
        self.game_over_message = ""

    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()

        while self.running:
            mouse_pos = pygame.mouse.get_pos()

            # Create card visuals for current room
            if not self.game_over:
                card_visuals = self.create_card_visuals()
                
                # Update button states
                self.avoid_button.disabled = not self.scoundrel.dungeon.can_avoid
                
                # Update hover states
                for card_visual in card_visuals:
                    card_visual.update_hover(mouse_pos)
                self.avoid_button.update_hover(mouse_pos)
            
            self.quit_button.update_hover(mouse_pos)

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.game_over:
                        pass  # Handle with keyboard in game over state
                    else:
                        # Check card clicks
                        for card_visual in card_visuals:
                            if card_visual.is_clicked(mouse_pos):
                                self.handle_card_interaction(card_visual.card)
                                break

                        # Check avoid button
                        if self.avoid_button.is_clicked(mouse_pos) and self.scoundrel.dungeon.can_avoid:
                            self.scoundrel.dungeon.avoid_room()

                        # Check quit button
                        if self.quit_button.is_clicked(mouse_pos):
                            self.running = False

                if event.type == pygame.KEYDOWN:
                    if self.game_over:
                        if event.key == pygame.K_r:
                            self.reset_game()
                        elif event.key == pygame.K_q:
                            self.running = False
                    else:
                        # Keyboard controls for cards
                        if event.key == pygame.K_1 and len(self.scoundrel.dungeon.current_room) > 0:
                            self.handle_card_interaction(self.scoundrel.dungeon.current_room[0])
                        elif event.key == pygame.K_2 and len(self.scoundrel.dungeon.current_room) > 1:
                            self.handle_card_interaction(self.scoundrel.dungeon.current_room[1])
                        elif event.key == pygame.K_3 and len(self.scoundrel.dungeon.current_room) > 2:
                            self.handle_card_interaction(self.scoundrel.dungeon.current_room[2])
                        elif event.key == pygame.K_4 and len(self.scoundrel.dungeon.current_room) > 3:
                            self.handle_card_interaction(self.scoundrel.dungeon.current_room[3])
                        elif event.key == pygame.K_a and self.scoundrel.dungeon.can_avoid:
                            self.scoundrel.dungeon.avoid_room()
                        elif event.key == pygame.K_q:
                            self.running = False

            # Drawing
            # screen.fill(DARK_GRAY)
            screen.blit(background, (0, 0))

            if not self.game_over:
                # Draw game elements
                self.draw_score(screen)
                self.draw_stats(screen)

                # Draw cards
                for card_visual in card_visuals:
                    card_visual.draw(screen)

                # Draw buttons
                self.avoid_button.draw(screen)
                self.quit_button.draw(screen)
                
                # Draw cards remaining under avoid button
                self.draw_cards_remaining(screen)

                # Draw instructions
                instructions = "Click cards or press 0-3 to interact | A to avoid room | Q to quit"
                inst_surface = small_font.render(instructions, True, LIGHT_GRAY)
                inst_rect = inst_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30))
                screen.blit(inst_surface, inst_rect)
            else:
                # Draw game over screen
                self.draw_game_over(screen)

            # Update display
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = ScoundrelGame()
    game.run()

