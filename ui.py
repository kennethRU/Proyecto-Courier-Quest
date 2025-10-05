# ui.py
import pygame
from typing import List, Tuple
from config import SCREEN_WIDTH, SCREEN_HEIGHT, STAMINA_MAX, REPUTATION_BONUS_THRESHOLD
from weather import weather_multiplier


def draw_hud(screen, font, player, weather_state, money, time_left, reputation, current_job_id, inv_list: List[str], mode_text: str):
    # Panel HUD
    pygame.draw.rect(screen, (20, 20, 20), (0, 0, SCREEN_WIDTH, 80))

    # Stamina
    stamina_ratio = max(0.0, min(1.0, player.stamina / STAMINA_MAX))
    pygame.draw.rect(screen, (60, 60, 60), (20, 20, 200, 20))
    pygame.draw.rect(screen, (80, 180, 80), (20, 20, int(200 * stamina_ratio), 20))
    screen.blit(font.render(f"Stamina: {int(player.stamina)}", True, (255, 255, 255)), (230, 18))

    # Reputación
    rep_color = (80, 180, 220) if reputation >= REPUTATION_BONUS_THRESHOLD else (220, 220, 80)
    screen.blit(font.render(f"Rep: {reputation}", True, rep_color), (20, 50))

    # Dinero
    screen.blit(font.render(f"$ {money}", True, (255, 255, 255)), (100, 50))

    # Tiempo
    screen.blit(font.render(f"Tiempo: {int(time_left)}s", True, (255, 255, 255)), (180, 50))

    # Clima
    wmul = weather_multiplier(weather_state.current)
    screen.blit(font.render(f"Clima: {weather_state.current} x{wmul:.2f}", True, (180, 200, 255)), (300, 50))

    # Inventario
    inv_text = f"INV {mode_text}: {', '.join(inv_list) if inv_list else '-'}"
    screen.blit(font.render(inv_text[:90], True, (255, 255, 255)), (20, 100))

    # Pedido actual
    if current_job_id:
        screen.blit(font.render(f"Actual: {current_job_id}", True, (255, 255, 255)), (20, 130))


def draw_message(screen, font, message: str):
    if not message:
        return
    surf = font.render(message, True, (255, 255, 255))
    screen.blit(surf, (20, SCREEN_HEIGHT - 40))
