<script setup lang="ts">
import { ref, reactive, onUnmounted, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { useSystemStore } from '../stores/system'

const systemStore = useSystemStore()
const config = reactive({
  confThreshold: 0.5,
  iouThreshold: 0.7,
  modelName: '',
  interval: 500 // 抓帧间隔(ms)
})

onMounted(async () => {
  await systemStore.fetchAvailableModels()
  config.modelName = systemStore.currentModel || systemStore.availableModels[0] || ''
})

const videoEl = ref<HTMLVideoElement | null>(null)
const canvasEl = ref<HTMLCanvasElement | null>(null)
const resultImage = ref('')
const running = ref(false)
const connecting = ref(false)

const stats = reactive({
  fps: 0,
  frameCount: 0,
  totalDamages: 0,
  severity: '无病害',
  severityScore: 0,
  cumulative: {} as Record<string, number>
})

let stream: MediaStream | null = null
let ws: WebSocket | null = null
let captureTimer: number | null = null
let waitingResponse = false

const severityTagType = (s: string) => {
  const m: Record<string, any> = {
    无病害: 'success', 轻微: 'success', 中等: 'warning', 严重: 'danger', 危险: 'danger'
  }
  return m[s] || 'info'
}

const captureFrame = (): string | null => {
  const video = videoEl.value
  const canvas = canvasEl.value
  if (!video || !canvas || video.videoWidth === 0) return null
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight
  const ctx = canvas.getContext('2d')
  if (!ctx) return null
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
  return canvas.toDataURL('image/jpeg', 0.7)
}

const start = async () => {
  connecting.value = true
  try {
    // 1. 打开摄像头
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 }
    })
    if (videoEl.value) {
      videoEl.value.srcObject = stream
      await videoEl.value.play()
    }

    // 2. 建立 WebSocket
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const wsUrl = `${proto}://${location.host}/api/detection/ws/realtime`
    ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      connecting.value = false
      running.value = true
      // 3. 定时抓帧发送
      captureTimer = window.setInterval(() => {
        if (!ws || ws.readyState !== WebSocket.OPEN || waitingResponse) return
        const frame = captureFrame()
        if (!frame) return
        waitingResponse = true
        ws.send(JSON.stringify({
          frame,
          conf: config.confThreshold,
          iou: config.iouThreshold,
          model: config.modelName
        }))
      }, config.interval)
    }

    ws.onmessage = (ev) => {
      waitingResponse = false
      const data = JSON.parse(ev.data)
      if (data.error) {
        ElMessage.error(data.error)
        return
      }
      resultImage.value = data.result_image
      stats.fps = data.fps
      stats.frameCount = data.frame_count
      stats.totalDamages = data.total_damages
      stats.severity = data.severity
      stats.severityScore = data.severity_score
      stats.cumulative = data.cumulative_counts || {}
    }

    ws.onerror = () => {
      ElMessage.error('WebSocket 连接错误')
      stop()
    }

    ws.onclose = () => {
      if (running.value) stop()
    }
  } catch (e: any) {
    connecting.value = false
    if (e.name === 'NotAllowedError') {
      ElMessage.error('摄像头权限被拒绝，请在浏览器中允许访问')
    } else if (e.name === 'NotFoundError') {
      ElMessage.error('未检测到摄像头设备')
    } else {
      ElMessage.error(e.message || '启动失败')
    }
    cleanup()
  }
}

const cleanup = () => {
  if (captureTimer !== null) {
    clearInterval(captureTimer)
    captureTimer = null
  }
  if (ws) {
    ws.onclose = null
    ws.close()
    ws = null
  }
  if (stream) {
    stream.getTracks().forEach(t => t.stop())
    stream = null
  }
  waitingResponse = false
}

const stop = () => {
  running.value = false
  cleanup()
}

const reset = () => {
  stop()
  resultImage.value = ''
  stats.fps = 0
  stats.frameCount = 0
  stats.totalDamages = 0
  stats.severity = '无病害'
  stats.severityScore = 0
  stats.cumulative = {}
}

onUnmounted(cleanup)
</script>

<template>
  <div class="page">
    <div class="page-header">
      <h1>实时检测</h1>
      <p>使用浏览器摄像头进行实时道路病害检测</p>
    </div>

    <el-row :gutter="24">
      <el-col :xs="24" :lg="8">
        <el-card shadow="hover" class="config-card">
          <template #header><span>检测配置</span></template>
          <el-form label-position="top">
            <el-form-item label="置信度阈值">
              <el-slider v-model="config.confThreshold" :min="0.1" :max="1" :step="0.05" :disabled="running" />
            </el-form-item>
            <el-form-item label="IOU 阈值">
              <el-slider v-model="config.iouThreshold" :min="0.1" :max="1" :step="0.05" :disabled="running" />
            </el-form-item>
            <el-form-item label="检测模型">
              <el-select v-model="config.modelName" style="width: 100%" :disabled="running">
                <el-option v-for="m in systemStore.availableModels" :key="m" :label="m" :value="m" />
              </el-select>
            </el-form-item>
            <el-form-item label="抓帧间隔 (ms)">
              <el-slider v-model="config.interval" :min="200" :max="2000" :step="100" :disabled="running" />
            </el-form-item>
            <el-form-item>
              <el-button v-if="!running" type="primary" :loading="connecting" @click="start" style="width: 100%">
                启动检测
              </el-button>
              <el-button v-else type="danger" @click="stop" style="width: 100%">停止检测</el-button>
              <el-button @click="reset" style="width: 100%; margin-top: 12px">重置</el-button>
            </el-form-item>
          </el-form>

          <!-- 实时统计 -->
          <div class="live-stats">
            <el-statistic title="FPS" :value="stats.fps" />
            <el-statistic title="累计帧数" :value="stats.frameCount" />
            <el-statistic title="当前帧病害" :value="stats.totalDamages" />
          </div>
          <div class="severity-line">
            当前严重度：
            <el-tag :type="severityTagType(stats.severity)">{{ stats.severity }}</el-tag>
            <span class="score">指数 {{ stats.severityScore.toFixed(1) }}</span>
          </div>
        </el-card>
      </el-col>

      <el-col :xs="24" :lg="16">
        <el-card shadow="hover" class="result-card">
          <template #header><span>实时画面</span></template>

          <!-- 隐藏的原始视频与抓帧画布 -->
          <video ref="videoEl" class="hidden-video" muted playsinline />
          <canvas ref="canvasEl" class="hidden-canvas" />

          <div v-if="!running && !resultImage" class="placeholder">
            <el-empty description="点击「启动检测」开启摄像头" />
          </div>

          <div v-else class="live-view">
            <img v-if="resultImage" :src="resultImage" class="live-image" alt="实时检测画面" />
            <div v-else class="waiting">
              <el-icon class="is-loading" :size="40"><Loading /></el-icon>
              <p>正在连接摄像头...</p>
            </div>

            <template v-if="Object.keys(stats.cumulative).length">
              <h3>累计病害统计</h3>
              <el-row :gutter="16">
                <el-col v-for="(count, type) in stats.cumulative" :key="type" :span="8">
                  <el-statistic :title="String(type)" :value="count" />
                </el-col>
              </el-row>
            </template>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.page { padding: 0; }
.page-header { margin-bottom: 24px; }
.page-header h1 { margin: 0 0 8px; font-size: 28px; color: #303133; }
.page-header p { margin: 0; color: #909399; }
.config-card, .result-card { border-radius: 12px; }
.live-stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-top: 8px;
  padding-top: 16px;
  border-top: 1px solid #ebeef5;
}
.severity-line {
  margin-top: 16px;
  color: #606266;
  font-size: 14px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.severity-line .score { color: #909399; }
.hidden-video, .hidden-canvas { display: none; }
.placeholder { padding: 40px 0; }
.live-image {
  width: 100%;
  border-radius: 8px;
  background: #000;
  display: block;
}
.waiting {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 60px 0;
  color: #909399;
}
.live-view h3 { margin: 24px 0 16px; font-size: 18px; color: #303133; }
@media (max-width: 992px) { .config-card { margin-bottom: 24px; } }
</style>
