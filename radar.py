import numpy as np
import cv2
import time
import math
import random

BLACK = (0,0,0)
WHITE = (255,255,255)
GREEN = (64,128,64)
RED = (0,0,255)
YELLOW = (0, 255,255)

class Map():
	def __init__(self):
		self.image = cv2.imread("map.png")
		self.height, self.width = self.image.shape[:2]

		# ドラゴンボールを埋める
		self.ball_cnt = 7
		self.ball_sides = [0] * self.ball_cnt		# ボールがレーダーのどちら側にあるか
		self.ball_states = [0] * self.ball_cnt		# レーダーがボールを検知したときの挙動
		self.ball_positions = []					# ボールの座標
		self.ball_positions.append((self.width//2, self.height//2))
		self.ball_positions.append((self.width//2+100, self.height//2+100))
		for _ in range(2, self.ball_cnt):
			x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
			self.ball_positions.append((x,y))

class Radar():
	def __init__(self, map):
		self.bai = 0				# 倍率の指数　0:1倍　1:2倍	-1:0.5倍
		self.mx, self.my = map.width//2, map.height//2		# 座標
		self.radius = 100			# 画像表示エリアのサイズは self.radius の2倍
		self.speed = 2				# レーダーが1周する時間
		self.tail_angle = 120		# レーダーの尾（軌跡）が残る角度
		self.tail_step = 10			# レーダーの尾（軌跡）のグラデーションのステップ数
		self.dir = 1				# 1:時計回り　-1:半時計回り
		self.circle_life = 50		# 光点の寿命

		self.map = map
		for x,y in map.ball_positions:
			cv2.circle(map.image, (x,y), 1, GREEN, -1)

		self.make_screen()
		self.get_map()

	def make_screen(self):
		size = 2 * self.radius
		self.trajectory_image = np.full((size,size,3), YELLOW, np.float32)	# レーダーの軌跡の初期値
		self.mask = np.full((size,size,3), BLACK, np.float32)	# マスクの初期値 0〜255の整数でなく0~1の小数
		n = 10
		self.back = np.full((size,size,3), GREEN, np.uint8)		# ドラゴンレーダーの背景
		for x in range(0, self.map.width, size//n):
			for y in range(0, self.map.width, size//n):
				cv2.line(self.back, (x,0), (x,self.map.height-1), BLACK, 1)
				cv2.line(self.back, (0,y), (self.map.width-1,y), BLACK, 1)

	def get_map(self, dx=0, dy=0, dk=0):
		"""
		変更された位置・倍率での地図のROIを取得する
		"""
		e = 10		# 移動量
		x, y = self.mx+e*dx, self.my+e*dy				# 移動後の座標（予定）
		self.bai += dk									# 倍率変更
		size = int(self.radius / 2 ** self.bai)			# マップ取得範囲の半分　場合によって２倍するのを忘れないこと

		# 中心座標が画面外に出ないようにする
		self.mx += e*dx if (0<x<self.map.width) else 0
		self.my += e*dy if (0<y<self.map.height) else 0

		# 左上座標と右下座標　画面外に出ないようにする
		x1, y1 = max(0, self.mx-size), max(0, self.my-size)
		x2, y2 = min(self.map.width, self.mx+size), min(self.map.height, self.my+size)

		# 指定したサイズでROIを取得
		roi_tmp = self.map.image[y1:y2, x1:x2]

		# 画面からはみ出ていたら周囲に枠を設定して本来のサイズにする
		if roi_tmp.shape[:2] == (2*size, 2*size):
			roi = roi_tmp
		else:
			x0 = 0 if self.mx > size else size-self.mx
			y0 = 0 if self.my > size else size-self.my
			tmp = np.full((2*size,2*size,3), GREEN, np.uint8)
			tmp[y0:y0+y2-y1, x0:x0+x2-x1] = roi_tmp
			roi = tmp

		# 表示用サイズに拡大縮小
		self.roi = cv2.resize(roi, (2*self.radius, 2*self.radius))


	def get_map0(self):
		"""
		指定した倍率で地図の一部を取得する
		広範囲 -> 広く取得して表示サイズまで縮小する
		詳細 -> 狭く取得して表示サイズまで拡大する
		画面外にはみ出さないようにする
		"""
		size = int(self.radius / 2 ** self.bai)
		x1, y1 = max(0, self.mx - size), max(0, self.my - size)
		x2, y2 = min(map.width, self.mx + size), min(map.height, self.my + size)
		roi = map.image[y1:y2, x1:x2]
		return cv2.resize(roi, (2*self.radius, 2*self.radius))
	
	def get_trajectory_mask(self, angle):
		"""
		軌跡のグラデーションのマスクを作成する
		np.uint8ではなくnp.float32で0~1の小数で作られているので、cv2.imshow()では確認することができない
		"""
		mask = self.mask.copy()
		unit_angle = 2 * math.pi * self.tail_angle / self.tail_step / 360		# グラデーションの一つの色の角度
		r = 1.1 * self.radius
		x0, y0 = self.radius, self.radius
		x1 = int(x0 + r * self.dir * math.cos(angle))
		y1 = int(y0 + r * self.dir * math.sin(angle))
		for i in range(self.tail_step):
			a = angle - self.dir * i * unit_angle
			x2 = int(x0 + r * self.dir * math.cos(a))
			y2 = int(y0 + r * self.dir * math.sin(a))
			c = (self.tail_step - i) / self.tail_step
			pts = np.array([(self.radius, self.radius), (x1,y1), (x2,y2)])
			cv2.fillConvexPoly(mask, pts, (c,c,c))
			x1, y1 = x2, y2
		return mask

	def find_ball(self, angle):
		"""
		ドラゴンボールを探す
		"""
		v1 = (math.cos(angle), math.sin(angle))							# レーダーの方向ベクトル
		for i in range(self.map.ball_cnt):
			x, y = self.map.ball_positions[i]
			v2 = (x-self.mx, y-self.my)									# ボールの相対位置ベクトル
			op = np.sign(np.cross(v1, v2))								# 外積の正負（レーダーのどちら側にあるか）
			ip = np.dot(v1, v2)											# 内積
			is_cross = self.dir * op * self.map.ball_sides[i]			# 現在の正負と一つ前の正負の乗算　レーダーを横切るとマイナスになる
			is_in_range = (0 <= ip <= self.radius / 2**self.bai)		# 内積がレーダー線分内にあるか
			if is_cross < 0 and is_in_range:
				# print(f"ball no.{i} is found! os={op}, is = {ip}")
				self.map.ball_states[i] = self.circle_life
			self.map.ball_sides[i] = op									# 現在の外積の正負を配列に格納する

	def draw_spotlight(self, screen):
		for i in range(self.map.ball_cnt):
			if self.map.ball_states[i] > 0:
				bx, by = self.map.ball_positions[i]
				x = self.radius + int((bx - self.mx) * 2**self.bai)
				y = self.radius + int((by - self.my) * 2**self.bai)
				r = self.circle_life - self.map.ball_states[i]

				# 少しずつ減衰する円を描画する
				alpha = self.map.ball_states[i]/self.circle_life						# アルファ値
				circle = np.full((2*self.radius,2*self.radius,3), BLACK, np.uint8)
				cv2.circle(circle, (x,y), r, WHITE, 2)									# 黒地に白の円（前景）
				masked_screen = (1/255 * screen * (255-circle)).astype(np.uint8)		# 背景の円の部分をマスクする
				compo_screen = cv2.addWeighted(screen, (1-alpha), circle, alpha, 0)		# 背景と前景を合成する　前景の黒地の部分も
				masked_circle = (1/255 * compo_screen * circle).astype(np.uint8)		# 合成画像の円以外の部分をマスクする
				screen = masked_screen + masked_circle									# 円をマスクした背景と背景をマスクした前景を合成する
				self.map.ball_states[i] -= 1
		return screen

	def show(self):
		sec = time.time() % 60										# 秒 0〜60
		angle = self.dir * 2 * math.pi * sec / self.speed			# 角度（ラジアン）
		self.find_ball(angle)										# ドラゴンボールを探す

		mask = self.get_trajectory_mask(angle)						# 軌跡のマスクを取得
		screen = self.draw_spotlight(self.back.copy())
		screen = (0.5 * screen + 0.5*self.roi).astype(np.uint8)
		screen = ((1 - mask) * screen + mask * self.trajectory_image).astype(np.uint8)
		
		cv2.imshow("rador", screen)

map = Map()
radar = Radar(map = map)

while True:
	radar.show()
	dx, dy, dk = 0, 0, 0					# キー入力で移動する方向と拡大縮小量
	key = cv2.waitKey(1)
	if key == 27:
		break
	elif key == ord("+"):
		dk = 1
	elif key == ord("-"):
		dk -= 1
	elif key == ord("a"):
		dx = 1
	elif key == ord("w"):
		dy = 1
	elif key == ord("s"):
		dy = -1
	elif key == ord("d"):
		dx = -1
	if not (dx==0 and dy==0 and dk==0):		# 移動もしくは拡大縮小したとき
		radar.get_map(dx, dy, dk)			# 地図表示エリア更新

	
cv2.destroyAllWindows()
