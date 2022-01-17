import numpy as np
import pygame as pg
from math import pi, atan, acos
import time

def rotate(surface, angle, pivot, offset):
    """Rotate the surface around the pivot point.

    Args:
        surface (pg.Surface): The surface that is to be rotated.
        angle (float): Rotate by this angle.
        pivot (tuple, list, pg.math.Vector2): The pivot point.
        offset (pg.math.Vector2): This vector is added to the pivot.
    """
    rotated_image = pg.transform.rotate(surface, -angle)  # Rotate the image.
    rotated_offset = offset.rotate(angle)  # Rotate the offset vector.
    # Add the offset vector to the center/pivot point to shift the rect.
    rect = rotated_image.get_rect(center=pivot+rotated_offset)
    return rotated_image, rect  # Return the rotated image and shifted rect.



r = 300
clock = pg.time.Clock()

screen = pg.display.set_mode((2*r, 2*r))
WHITE=(255,255,255)
RED=(255,0,0)
GREEN=(0, 255, 0)
YELLOW=(255, 255, 0)
BLUE=(0,0,255)
BLACK=(0,0,0)
screen.fill(WHITE)
pg.draw.circle(screen, BLACK, (r,r), r, width=20)
v = pg.math.Vector2(50, 50)
center = pg.math.Vector2(r, r)


arm_mats = np.zeros((12, 4, 4))
arm_mats[:,:,:] = np.identity(4)
arm_mat = arm_mats[0]
arm_mat[0][3] = 2 * r
arm_0_rotation = np.identity(4)
degree = 1
theta = np.radians(degree)
c, s = np.cos(theta), np.sin(theta)
arm_0_rotation[:2, :2] = np.array(((c, -s), (s, c)))

leng = 2**.5 * r
# [[base_x, tip_x]
#  [base_y, tip_y]]
arm_vecs = np.array([[0, leng],[0, 0], [0,0],[1,1]])
get_to_center = np.identity(4)
get_to_center[0,3] = r
get_to_center[1,3] = r

pygame = pg
arm_box = (2**.5 * r, r * 2 * (1 - (2**.5 / 2)))
arm = pygame.Surface(arm_box)
arm.set_colorkey(WHITE)
arm.fill(WHITE)
pygame.draw.arc(arm, BLACK, [(2*r - arm_box[0])/-2,0,2*r,2*r], 0, pi, 5)#[0,0,2*r,2*r]
arm.blit(pygame.transform.flip(arm, 0, 1), (0,0), None, 9)

empty_arm = pygame.Surface(arm_box)
empty_arm.set_colorkey(WHITE)
empty_arm.fill(WHITE)

step = np.identity(4)
degree = 1
theta = np.radians(degree)
c, s = np.cos(theta), np.sin(theta)
step[:2, :2] = np.array(((c, -s), (s, c))) # rotate 1 degree
step = step.dot(get_to_center)

def add_arm(blits, arm_vecs):
    arm_ind = len(blits)
    tip = arm_vecs[:2,1]
    end = arm_vecs[:2,0]
    diff = tip - end
    cneter = to_pygame((tip + end) / 2)
    angle = 180 * atan(diff[1] / diff[0]) / pi
    arm_rot = pg.transform.rotate(arm, angle)
    rect = arm_rot.get_rect(center=cneter)
    blits.append((arm_rot, rect, None, 9))
    pygame.draw.line(screen, BLACK, to_pygame(tip), to_pygame(points_to_run_string_to[arm_ind]))
    return blits

def draw_arms(blits):
    return screen.blits(blits)

def draw_arm(arm_vecs, c=BLACK):
    tip = arm_vecs[:2,1]
    end = arm_vecs[:2,0]
    diff = tip - end
    cneter = to_pygame((tip + end) / 2)
    angle = 180 * atan(diff[1] / diff[0]) / pi
    arm_rot = pg.transform.rotate(arm, angle)
    rect = arm_rot.get_rect(center=cneter)
    screen.blit(arm_rot, rect, None, 9)

def erase_arm(arm_vecs):
    tip = arm_vecs[:2,1]
    end = arm_vecs[:2,0]
    diff = tip - end
    tip = to_pygame(tip)
    end = to_pygame(end)
    angle = 180 * atan(diff[1] / diff[0]) / pi
    arm_rot = pg.transform.rotate(empty_arm, angle)
    screen.fill(WHITE, arm_rot.get_rect(topleft=(min(tip[0], end[0])-50, min(tip[1], end[1])-50)))
    #screen.blit(arm_rot, (min(tip[0], end[0]), min(tip[1], end[1])), None, 9)

# TO_D * P0 = PD
#

def erase_arms():
    pygame.draw.circle(screen, WHITE, (r,r), r)

def mat_for_degree(degree):
    step = np.identity(4)
    theta = np.radians(degree)
    c, s = np.cos(theta), np.sin(theta)
    step[:2, :2] = np.array(((c, -s), (s, c)))
    return step

def mat_for_trans(x, y):
    step = np.identity(4)
    step[0,3] = x
    step[1,3] = y
    return step

def compute_T_for_arm(angle, ind, count, r):
    '''
    :param vectors:
    :param angle:
    :param ind:
    :param count:
    :param r:
    :return:
    '''
    #return mat_for_degree(0)
    #T0 = mat_for_trans(2*r, r) # translate 0,0 to 2r, r
    T0 = mat_for_degree(ind * 360 / count) # rotate to correct angle
    T1 = mat_for_trans(r, 0) # rotate to correct angle
    T2 = mat_for_degree(angle)
    return T2.dot(T1.dot(T0))

def to_pygame(coords, height=2*r):
    """Convert coordinates into pygame coordinates (lower-left => top left)."""
    return (coords[0], height - coords[1])
trans_rot_cent_old = None
def translate_to_be_displayed(r):
    return mat_for_trans(r, r)

def mat_to_point(mat):
    return mat[0:2,-1]

arm_vecs = np.array([[0, leng],[0, 0], [0,0],[1,1]])
running = True
print(mat_for_degree(45))
print(arm_vecs)
dots_count = 1
dots = [mat_for_trans(r,r)] * dots_count
count=3*dots_count
consts = [mat_for_trans(r,r).dot(mat_for_degree(360*i/count).dot(mat_for_trans(r, 0))) for i in range(count)]
olds = []
screen.blit(arm, (100,100), None, 9)

points_to_run_string_to = [const.dot(mat_for_trans(-r,0).dot(mat_for_degree(90)).dot(mat_for_trans(r,0)))[0:2,-1] for const in consts]

diffs = [-1] * count
#deltas = [min(int(90 * i /count), 90) for i in range(count)]
deltas = [1] * count
frame_rate = 60
lag_const = 1#2 * frame_rate
circle_center = np.array([r,r])

dot_radius = 25

dot_point = pygame.mouse.get_pos()
delta_from_center = dot_point-circle_center
dis_from_center = np.linalg.norm(delta_from_center)
if dis_from_center > (r - dot_radius):
    dot_point = circle_center + (delta_from_center * (r - dot_radius) / dis_from_center)

dotlag = [dot_point] * lag_const

user_controlled = False
t = 0


rot_speed = .666
while running:
    t += 1
    clock.tick(frame_rate)
    if user_controlled:
        dot_point = pygame.mouse.get_pos()
    else:
        dot_point = (r + .5* r * (np.sin(rot_speed* pi * t / frame_rate)), r + .5 * r * (np.sin(rot_speed*pi * 2*t / frame_rate)))
    delta_from_center = dot_point-circle_center
    dis_from_center = np.linalg.norm(delta_from_center)
    if dis_from_center > (r - dot_radius):
        dot_point = circle_center + (delta_from_center * (r - dot_radius) / dis_from_center)

    dotlag.append(dot_point)
    dotlag = dotlag[1:]
    for i in range(count):
        #if deltas[i] % 90 == 0:
        #    diffs[i] = diffs[i] * -1
        #deltas[i] = deltas[i] + diffs[i]
        ptrst = to_pygame(points_to_run_string_to[i])
        ptrst = np.array([ptrst[0], ptrst[1]])
        #dot_point = mat_to_point(dots[i % dots_count])
        dot_point = dotlag[int(len(dotlag) * (i % dots_count) / dots_count)]
        if i==7:
            print(dot_point)
        dist = min(np.linalg.norm(ptrst - dot_point), 2*r)
        pg.draw.circle(screen, BLACK, dot_point, 3)
        deltas[i] = 180 * acos(1-((2*r-dist)**2/(4*r*r))) / np.pi#45 * (np.sin(np.radians(deg_c + 360 * i / count)) + 1)
        #print(dot_point)
        #print(ptrst)
        #print(dist)
        #print(deltas[i])
        #print()
    #print(deltas)
    #print("deltas"+str(deltas))
    # Did the user click the window close button?
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                user_controlled = not user_controlled
    #clock.tick(60)
    #pg.draw.circle(screen, WHITE, v + center, 5, width=10)
    #v = v.rotate(1)
    #pg.draw.circle(screen, BLACK, v + center, 5, width=10)
    #erase_arm(arm_vecs)
    #draw_arm(arm_vecs, GREEN)

    #draw_arm(translate_to_be_displayed(r).dot(arm_vecs))
    #draw_arm(compute_T_for_arm(45,1,12,r).dot(translate_to_be_displayed(r).dot(arm_vecs)), RED)
    #draw_arm(arm_vecs, GREEN)
    #arm_centered = mat_for_trans(r,r).dot(arm_vecs)
    #arm_rot = mat_for_trans(r,r).dot(mat_for_degree(-90).dot(arm_vecs))
    #draw_arm(arm_centered)
    #arm_rot = mat_for_trans(r,r).dot(mat_for_degree(90)).dot(arm_vecs)
    #draw_arm(arm_rot, RED)#draw_arm(mat_for_degree(90).dot(arm_vecs), RED)
    #draw_arm(mat_for_trans(50,50).dot(mat_for_degree(90).dot(arm_vecs)), GREEN)
    #draw_arm(mat_for_degree(90).dot(mat_for_trans(50,50).dot(mat_for_degree(90).dot(arm_vecs))), BLUE)
    #draw_arm(arm_vecs)
    #centered_arm = mat_for_trans(r,r).dot(arm_vecs)
    #draw_arm(centered_arm, RED)
    #rot_cent_arm = mat_for_trans(r,r).dot(mat_for_degree(30)).dot(arm_vecs)
    #draw_arm(rot_cent_arm, GREEN)
    #if len(olds) > 0:
    #    for trans_rot_cent_old in olds:
    #        erase_arm(trans_rot_cent_old)
    erase_arms()
    arms = []
    for i, const in enumerate(consts):
        arms.append(const.dot(mat_for_degree(135 + deltas[i]).dot(arm_vecs)))
    blits = []
    for trans_rot_cent in arms:
        blits = add_arm(blits, trans_rot_cent)
    draw_arms(blits)
    #draw_arm(trans_rot_cent, BLUE)
    #print(mat_for_degree(45).dot(arm_vecs))
    #draw_arm(mat_for_trans(r,r).dot(mat_for_degree(45).dot(arm_vecs)), YELLOW)
    #draw_arm(mat_for_degree(45).dot(mat_for_trans(r,r).dot(arm_vecs)), BLUE)
    #print(mat_for_degree(45).dot(mat_for_trans(100,100).dot(arm_vecs)))
    pg.draw.circle(screen, BLACK, (r,r), r, width=20)
    #pg.draw.circle(screen, RED, (r,r), r/2, width=1)
    for dot_index in range(dots_count):
        dot_point = dotlag[int(len(dotlag) * dot_index / dots_count)]
        pg.draw.circle(screen, BLACK, dot_point, dot_radius)
        for i in range(count):
            if i % dots_count != dot_index:
                continue
            if (i==7):
                print(dot_point)
                print()
            pg.draw.line(screen, BLACK, to_pygame(points_to_run_string_to[i]), dot_point, 1)

    pg.display.flip()


pg.quit()
