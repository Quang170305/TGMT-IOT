import pygame
import random
import time
import threading
import cv2
import mediapipe as mp
import numpy as np
from pygame.locals import *

# ========================
# CÁC HẰNG SỐ GAME
# ========================
SCREEN_WIDTH  = 400
SCREEN_HEIGHT = 600
SPEED         = 20
GRAVITY       = 2.5
GAME_SPEED    = 15
PIPE_GAP      = 150
PIPE_WIDTH    = 80
PIPE_HEIGHT   = 500
GROUND_WIDTH  = 2 * SCREEN_WIDTH
GROUND_HEIGHT = 100

# ========================
# MEDIAPIPE – HAND DETECTION (chạy thread riêng)
# ========================
fist_detected  = False  # biến chia sẻ giữa thread camera và game
swipe_detected = False  # phát hiện quét tay xuống

def is_hand_closed(lm):
    """Trả về True nếu ít nhất 3/4 ngón tay gập vào."""
    pts = [(l.x, l.y) for l in lm.landmark]
    closed = sum([
        pts[8][1]  > pts[6][1],   # ngón trỏ
        pts[12][1] > pts[10][1],  # ngón giữa
        pts[16][1] > pts[14][1],  # ngón áp út
        pts[20][1] > pts[18][1],  # ngón út
    ])
    return closed >= 3

# ========================
# TẠO ÂM THANH TỪ FILE THẬT
# ========================
def make_wing_sound():
    return pygame.mixer.Sound('assets/audio/wing.wav')

def make_hit_sound():
    return pygame.mixer.Sound('assets/audio/hit.wav')


def camera_thread():
    global fist_detected, swipe_detected
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1
    )
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    time.sleep(1.5)

    # Lưu lịch sử Y cổ tay để phát hiện quét tay
    wrist_y_history = []
    HISTORY_LEN  = 6   # số frame để phân tích
    SWIPE_THRESH = 0.09 # ngưỡng quét (đơn vị tọa độ chuẩn hóa 0-1)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = hands.process(rgb)
        rgb.flags.writeable = True
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = frame.shape[:2]

        fist_now  = False
        swipe_now = False

        if result.multi_hand_landmarks:
            for hl in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

                # ---- Phát hiện NAM TAY ----
                if is_hand_closed(hl):
                    fist_now = True

                # ---- Phát hiện QUÉT TAY XUỐNG ----
                wrist_y = hl.landmark[0].y  # 0 = WRIST, tọa độ 0-1
                wrist_y_history.append(wrist_y)
                if len(wrist_y_history) > HISTORY_LEN:
                    wrist_y_history.pop(0)

                if len(wrist_y_history) == HISTORY_LEN:
                    delta = wrist_y_history[-1] - wrist_y_history[0]
                    if delta > SWIPE_THRESH:   # tay di chuyển xuống nhanh
                        swipe_now = True

                # Vẽ trail quét tay: vẽ chấm lịch sử cổ tay lên camera
                for i, wy in enumerate(wrist_y_history):
                    alpha = int(255 * (i + 1) / HISTORY_LEN)
                    cx = int(hl.landmark[0].x * w)
                    cy = int(wy * h)
                    cv2.circle(frame, (cx, cy), 5, (0, alpha, 255 - alpha), -1)
        else:
            wrist_y_history.clear()

        fist_detected  = fist_now
        swipe_detected = swipe_now

        # Overlay hiển thị trạng thái
        if swipe_now:
            label, color = "SWIPE -> NHAY!", (255, 80, 0)
        elif fist_now:
            label, color = "NAM TAY -> NHAY!", (0, 0, 255)
        else:
            label, color = "xoe tay / quet xuong...", (0, 220, 80)

        disp = cv2.flip(frame, 1)
        cv2.putText(disp, label, (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        cv2.imshow("Camera (ESC de thoat)", disp)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ========================
# SPRITES GAME
# ========================
def make_color_surface(w, h, color):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    s.fill(color)
    return s

class Bird(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.images = [
            pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha(),
            pygame.image.load('assets/sprites/bluebird-downflap.png').convert_alpha(),
        ]
        self.speed = SPEED
        self.current_image = 0
        self.image = self.images[0]
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect.x = SCREEN_WIDTH // 6
        self.rect.y = SCREEN_HEIGHT // 2

    def update(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        self.mask = pygame.mask.from_surface(self.image)
        self.speed += GRAVITY
        self.rect.y += int(self.speed)

    def bump(self, wing_snd=None):
        self.speed = -SPEED
        if wing_snd:
            wing_snd.play()

    def begin(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]

class Pipe(pygame.sprite.Sprite):
    def __init__(self, inverted, xpos, ysize):
        super().__init__()
        self.image = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDTH, PIPE_HEIGHT))
        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect[1] = -(self.rect[3] - ysize)
        else:
            self.rect[1] = SCREEN_HEIGHT - ysize
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        self.rect.x -= GAME_SPEED

class Ground(pygame.sprite.Sprite):
    def __init__(self, xpos):
        super().__init__()
        self.image = pygame.image.load('assets/sprites/base.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDTH, GROUND_HEIGHT))
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect(x=xpos, y=SCREEN_HEIGHT - GROUND_HEIGHT)

    def update(self):
        self.rect.x -= GAME_SPEED

def is_off_screen(sprite):
    return sprite.rect.right < 0

def get_random_pipes(xpos):
    size = random.randint(100, 300)
    return (Pipe(False, xpos, size),
            Pipe(True,  xpos, SCREEN_HEIGHT - size - PIPE_GAP))



# ========================
# VÒNG LẶP CHÍNH
# ========================
def main():
    global fist_detected, swipe_detected

    # Khởi động camera thread riêng biệt
    t = threading.Thread(target=camera_thread, daemon=True)
    t.start()

    pygame.init()
    pygame.mixer.init(44100, -16, 1, 512)
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird – Hand Controller")
    clock = pygame.time.Clock()

    # Tạo âm thanh procedural
    wing_snd = make_wing_sound()
    hit_snd  = make_hit_sound()

    font_big   = pygame.font.SysFont("Arial", 48, bold=True)
    font_small = pygame.font.SysFont("Arial", 26)

    # Load ảnh nền
    sky = pygame.image.load('assets/sprites/background-day.png').convert()
    sky = pygame.transform.scale(sky, (SCREEN_WIDTH, SCREEN_HEIGHT))
    begin_img   = pygame.image.load('assets/sprites/message.png').convert_alpha()
    gameover_img = pygame.image.load('assets/sprites/gameover.png').convert_alpha()

    # Sprite groups
    bird_group   = pygame.sprite.Group()
    bird         = Bird()
    bird_group.add(bird)

    ground_group = pygame.sprite.Group()
    for i in range(2):
        ground_group.add(Ground(GROUND_WIDTH * i))

    pipe_group = pygame.sprite.Group()
    for i in range(2):
        for p in get_random_pipes(SCREEN_WIDTH * i + 800):
            pipe_group.add(p)

    score = 0
    prev_fist = False

    # ---- Màn hình chờ ----
    waiting = True
    while waiting:
        clock.tick(15)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); return
            if event.type == KEYDOWN and event.key in (K_SPACE, K_UP):
                bird.bump(wing_snd); waiting = False

        # Kích hoạt bằng nắm tay hoặc quét tay
        trigger = (fist_detected and not prev_fist) or swipe_detected
        if trigger:
            bird.bump(wing_snd); waiting = False
        prev_fist = fist_detected

        screen.blit(sky, (0, 0))
        ground_group.update()
        bird.begin()

        if is_off_screen(ground_group.sprites()[0]):
            ground_group.remove(ground_group.sprites()[0])
            ground_group.add(Ground(GROUND_WIDTH - 20))

        bird_group.draw(screen)
        ground_group.draw(screen)

        # Hiển thị message.png gốc của game
        screen.blit(begin_img, begin_img.get_rect(center=(SCREEN_WIDTH//2, 250)))

        # Trạng thái tay (góc trên trái)
        status_color = (80, 255, 80) if fist_detected else (255, 255, 255)
        hand_txt = font_small.render(
            "!NAM TAY!" if fist_detected else "Nam tay / Space", True, status_color)
        screen.blit(hand_txt, (10, 10))

        pygame.display.update()

    # ---- Màn hình chơi ----
    running = True
    while running:
        clock.tick(15)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); return
            if event.type == KEYDOWN and event.key in (K_SPACE, K_UP):
                bird.bump(wing_snd)

        # Phát hiện nắm tay (cạnh sườn) HOẶC quét tay
        trigger = (fist_detected and not prev_fist) or swipe_detected
        if trigger:
            bird.bump(wing_snd)
        prev_fist = fist_detected

        # Cập nhật
        bird_group.update()
        ground_group.update()
        pipe_group.update()

        # Cuộn mặt đất
        if is_off_screen(ground_group.sprites()[0]):
            ground_group.remove(ground_group.sprites()[0])
            ground_group.add(Ground(GROUND_WIDTH - 20))

        # Tạo ống mới
        if is_off_screen(pipe_group.sprites()[0]):
            pipe_group.remove(pipe_group.sprites()[0])
            pipe_group.remove(pipe_group.sprites()[0])
            for p in get_random_pipes(SCREEN_WIDTH * 2):
                pipe_group.add(p)
            score += 1

        # Vẽ
        screen.blit(sky, (0, 0))
        pipe_group.draw(screen)
        ground_group.draw(screen)
        bird_group.draw(screen)

        # HUD điểm
        score_surf = font_big.render(str(score), True, (255, 255, 255))
        screen.blit(score_surf, score_surf.get_rect(center=(SCREEN_WIDTH//2, 60)))

        # Trạng thái tay góc trên trái
        status_color = (80, 255, 80) if fist_detected else (200, 200, 200)
        hand_txt = font_small.render(
            "NAM TAY" if fist_detected else "xoe tay...", True, status_color)
        screen.blit(hand_txt, (10, 10))

        pygame.display.update()

        # Va chạm
        if (pygame.sprite.groupcollide(bird_group, ground_group, False, False,
                                       pygame.sprite.collide_mask) or
            pygame.sprite.groupcollide(bird_group, pipe_group,   False, False,
                                       pygame.sprite.collide_mask)):
            hit_snd.play()
            running = False

    # ---- Game Over ----
    for _ in range(30):
        clock.tick(15)
        screen.blit(sky, (0, 0))
        pipe_group.draw(screen); ground_group.draw(screen); bird_group.draw(screen)
        screen.blit(gameover_img, gameover_img.get_rect(center=(SCREEN_WIDTH//2, 260)))
        sc = font_small.render(f"Score: {score}", True, (255, 255, 255))
        screen.blit(sc, sc.get_rect(center=(SCREEN_WIDTH//2, 330)))
        pygame.display.update()

    time.sleep(1.5)
    # Đợi nhấn lại hoặc nắm tay lại để chơi tiếp
    waiting_restart = True
    prev_fist = fist_detected
    while waiting_restart:
        clock.tick(15)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); return
            if event.type == KEYDOWN and event.key in (K_SPACE, K_UP, K_r):
                waiting_restart = False
        if fist_detected and not prev_fist:
            waiting_restart = False
        prev_fist = fist_detected

        screen.blit(sky, (0, 0))
        screen.blit(gameover_img, gameover_img.get_rect(center=(SCREEN_WIDTH//2, 220)))
        sc  = font_small.render(f"Score: {score}", True, (255, 255, 255))
        rst = font_small.render("Nam tay / Space de choi lai", True, (255, 240, 100))
        screen.blit(sc,  sc.get_rect(center=(SCREEN_WIDTH//2, 290)))
        screen.blit(rst, rst.get_rect(center=(SCREEN_WIDTH//2, 340)))
        pygame.display.update()

    main()  # Chơi lại

if __name__ == "__main__":
    main()