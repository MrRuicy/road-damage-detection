<script setup lang="ts">
import { ref, reactive, onUnmounted, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { detectVideo, getVideoTask, type VideoTaskStatus, type VideoTaskResult } from '../api/video'
import { useSystemStore } from '../stores/system'

const systemStore = useSystemStore()
const fileList = ref<any[]>([])
const config = reactive({
  confThreshold: 0.5,
  iouThreshold: 0.7,
  modelName: '',
  frameSkip: 1,
  downscale: 1.0
})

onMounted(async () => {
  await systemStore.fetchAvailableModels()
  config.modelName = systemStore.currentModel || systemStore.availableModels[0] || ''
})

const processing = ref(false)
const progress = ref(0)
const statusMsg = ref('')
const result = ref<VideoTaskResult | null>(null)
let pollTimer: number | null = null

const handleFileChange = (file: any) => {
  fileList.value = [file]
}

const handleRemove = () => {
  fileList.value = []
}

const clearTimer = () => {
  if (pollTimer !== null) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

const poll = (taskId: string) => {
  pollTimer = window.setInterval(async () => {
    try {
      const task: VideoTaskStatus = await getVideoTask(taskId)
      progress.value = task.progress
      statusMsg.value = task.message

      if (task.status === 'completed') {
        clearTimer()
        processing.value = false
        result.value = task.result || null
        ElMessage.success('视频检测完成')
      } else if (task.status === 'failed') {
        clearTimer()
        processing.value = false
        ElMessage.error(task.error || '视频检测失败')
      }
    } catch (e: any) {
      clearTimer()
      processing.value = false
      ElMessage.error(e.message || '查询任务失败')
    }
  }, 1500)
}

const handleDetect = async () => {
  if (fileList.value.length === 0) {
    ElMessage.warning('请先上传视频')
    return
  }
  result.value = null
  processing.value = true
  progress.value = 0
  statusMsg.value = '正在上传...'

  try {
    const formData = new FormData()
    formData.append('file', fileList.value[0].raw as File)
    formData.append('conf_threshold', String(config.confThreshold))
    formData.append('iou_threshold', String(config.iouThreshold))
    formData.append('model_name', config.modelName)
    formData.append('frame_skip', String(config.frameSkip))
    formData.append('downscale', String(config.downscale))

    const { task_id } = await detectVideo(formData)
    statusMsg.value = '开始检测...'
    poll(task_id)
  } catch (e: any) {
    processing.value = false
    ElMessage.error(e.message || '上传失败')
  }
}

const handleReset = () => {
  clearTimer()
  fileList.value = []
  result.value = null
  processing.value = false
  progress.value = 0
  statusMsg.value = ''
}

const getSeverityType = (severity: string) => {
  const types: Record<string, any> = {
    无病害: 'success', 轻微: 'success', 中等: 'warning', 严重: 'danger', 危险: 'danger'
  }
  return types[severity] || 'info'
}

onUnmounted(clearTimer)
</script>

<template>
  <div class="page">
    <div class="page-header">
      <h1>视频检测</h1>
      <p>上传道路视频进行逐帧病害检测</p>
    </div>

    <el-row :gutter="24">
      <el-col :xs="24" :lg="8">
        <el-card shadow="hover" class="config-card">
          <template #header><span>检测配置</span></template>
          <el-form label-position="top" class="compact-form">
            <el-row :gutter="16">
              <el-col :span="12">
                <el-form-item :label="`置信度 ${config.confThreshold}`">
                  <el-slider v-model="config.confThreshold" :min="0.1" :max="1" :step="0.05" size="small" />
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item :label="`IOU ${config.iouThreshold}`">
                  <el-slider v-model="config.iouThreshold" :min="0.1" :max="1" :step="0.05" size="small" />
                </el-form-item>
              </el-col>
            </el-row>
            <el-row :gutter="16">
              <el-col :span="12">
                <el-form-item :label="`跳帧 ${config.frameSkip}`">
                  <el-slider v-model="config.frameSkip" :min="0" :max="10" :step="1" size="small" />
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item :label="`缩放 ${config.downscale}`">
                  <el-slider v-model="config.downscale" :min="0.3" :max="1" :step="0.1" size="small" />
                </el-form-item>
              </el-col>
            </el-row>
            <el-form-item label="检测模型">
              <el-select v-model="config.modelName" style="width: 100%" size="small">
                <el-option v-for="m in systemStore.availableModels" :key="m" :label="m" :value="m" />
              </el-select>
            </el-form-item>
            <el-form-item label="上传视频">
              <el-upload
                drag
                :limit="1"
                :auto-upload="false"
                :file-list="fileList"
                :on-change="handleFileChange"
                :on-remove="handleRemove"
                accept="video/*"
                class="compact-upload"
                style="width: 100%"
              >
                <el-icon class="upload-icon"><VideoCamera /></el-icon>
                <div class="upload-text">拖拽视频或点击上传 · MP4/AVI/MOV</div>
              </el-upload>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :loading="processing" @click="handleDetect" style="width: 100%">
                开始检测
              </el-button>
              <el-button @click="handleReset" style="width: 100%; margin-top: 12px">重置</el-button>
            </el-form-item>
          </el-form>
        </el-card>
      </el-col>

      <el-col :xs="24" :lg="16">
        <el-card shadow="hover" class="result-card">
          <template #header><span>检测结果</span></template>

          <!-- 处理进度 -->
          <div v-if="processing" class="progress-box">
            <el-progress type="dashboard" :percentage="Math.round(progress)" :width="160" />
            <p class="status-msg">{{ statusMsg }}</p>
          </div>

          <el-empty v-else-if="!result" description="暂无检测结果" />

          <!-- 结果 -->
          <div v-else class="result-box">
            <el-alert
              :title="`严重程度: ${result.severity}（指数 ${result.severity_score.toFixed(1)}）`"
              :type="getSeverityType(result.severity)"
              show-icon :closable="false" style="margin-bottom: 16px"
            >
              检测 {{ result.processed_frames }}/{{ result.total_frames }} 帧，发现 {{ result.total_damages }} 处病害，耗时 {{ result.processing_time }}s
            </el-alert>

            <video
              :src="result.result_video_url"
              controls
              class="result-video"
            />

            <el-descriptions :column="2" border style="margin-top: 16px">
              <el-descriptions-item label="视频名称">{{ result.video_name }}</el-descriptions-item>
              <el-descriptions-item label="帧率">{{ result.fps }} fps</el-descriptions-item>
              <el-descriptions-item label="总帧数">{{ result.total_frames }}</el-descriptions-item>
              <el-descriptions-item label="检测帧数">{{ result.processed_frames }}</el-descriptions-item>
            </el-descriptions>

            <template v-if="Object.keys(result.class_counts).length">
              <h3>病害统计</h3>
              <el-row :gutter="16">
                <el-col v-for="(count, type) in result.class_counts" :key="type" :span="8">
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
.page-header h1 { margin: 0 0 8px; font-size: 28px; }
.page-header p { margin: 0; }
.config-card, .result-card { border-radius: 12px; }
.compact-form :deep(.el-form-item) { margin-bottom: 10px; }
.compact-form :deep(.el-form-item__label) { padding-bottom: 2px; line-height: 1.4; }
.compact-upload :deep(.el-upload-dragger) { padding: 16px 10px; }
.upload-icon { font-size: 36px; color: #409eff; margin-bottom: 8px; }
.upload-text { font-size: 13px; color: #606266; }
.progress-box { display: flex; flex-direction: column; align-items: center; padding: 40px 0; }
.status-msg { margin-top: 20px; color: #606266; }
.result-video { width: 100%; max-height: 420px; border-radius: 8px; background: #000; }
.result-box h3 { margin: 24px 0 16px; font-size: 18px; }
@media (max-width: 992px) { .config-card { margin-bottom: 24px; } }
</style>
