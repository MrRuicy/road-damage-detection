<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { ElMessage, ElLoading } from 'element-plus'
import { detectImage, detectBatch } from '../api/detection'
import type { DetectionResult, BatchDetectionResult } from '../api/detection'
import type { UploadUserFile } from 'element-plus'
import { useSystemStore } from '../stores/system'

const systemStore = useSystemStore()
const detectionMode = ref<'single' | 'batch'>('single')
const fileList = ref<UploadUserFile[]>([])
const config = reactive({
  confThreshold: 0.5,
  iouThreshold: 0.7,
  modelName: ''
})

onMounted(async () => {
  await systemStore.fetchAvailableModels()
  config.modelName = systemStore.currentModel || systemStore.availableModels[0] || ''
})

const singleResult = ref<DetectionResult | null>(null)
const batchResult = ref<BatchDetectionResult | null>(null)
const originalPreview = ref<string>('')
const loading = ref(false)

const handleFileChange = (file: any) => {
  fileList.value = [file]
}

const handleBatchChange = (_file: any, files: any[]) => {
  fileList.value = files
}

const handleRemove = (_file: any, files: any[]) => {
  fileList.value = files
}

const handleDetect = async () => {
  if (fileList.value.length === 0) {
    ElMessage.warning('请先上传图片')
    return
  }

  loading.value = true
  const loadingInstance = ElLoading.service({ fullscreen: true, text: '检测中...' })

  try {
    const formData = new FormData()

    if (detectionMode.value === 'single') {
      formData.append('file', fileList.value[0].raw as File)
      formData.append('conf_threshold', String(config.confThreshold))
      formData.append('iou_threshold', String(config.iouThreshold))
      formData.append('model_name', config.modelName)

      // 捕获原图预览
      originalPreview.value = URL.createObjectURL(fileList.value[0].raw as File)

      const result = await detectImage(formData)
      singleResult.value = result
      batchResult.value = null
      if (result.is_duplicate) {
        ElMessage.info('该图片在相同模型与参数下已检测过，已复用历史结果')
      } else {
        ElMessage.success('检测完成')
      }
    } else {
      fileList.value.forEach(file => {
        formData.append('files', file.raw as File)
      })
      formData.append('conf_threshold', String(config.confThreshold))
      formData.append('iou_threshold', String(config.iouThreshold))
      formData.append('model_name', config.modelName)

      const result = await detectBatch(formData)
      batchResult.value = result
      singleResult.value = null
      ElMessage.success(`批量检测完成，共检测 ${result.total_images} 张图片`)
    }
  } catch (error: any) {
    ElMessage.error(error.message || '检测失败')
  } finally {
    loading.value = false
    loadingInstance.close()
  }
}

const handleReset = () => {
  fileList.value = []
  singleResult.value = null
  batchResult.value = null
}

const getSeverityType = (severity: string) => {
  const types: Record<string, any> = {
    '无病害': 'success',
    '轻微': 'success',
    '中等': 'warning',
    '严重': 'danger',
    '危险': 'danger'
  }
  return types[severity] || 'info'
}
</script>

<template>
  <div class="detection-page">
    <div class="page-header">
      <h1>图像检测</h1>
      <p>上传道路图像进行病害检测</p>
    </div>

    <el-row :gutter="24">
      <!-- 左侧配置区 -->
      <el-col :xs="24" :lg="8">
        <el-card shadow="hover" class="config-card">
          <template #header>
            <span>检测配置</span>
          </template>

          <el-form label-position="top" class="compact-form">
            <el-row :gutter="16">
              <el-col :span="12">
                <el-form-item label="检测模式">
                  <el-radio-group v-model="detectionMode" @change="handleReset" size="small">
                    <el-radio-button value="single">单张</el-radio-button>
                    <el-radio-button value="batch">批量</el-radio-button>
                  </el-radio-group>
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item label="检测模型">
                  <el-select v-model="config.modelName" style="width: 100%" size="small">
                    <el-option v-for="m in systemStore.availableModels" :key="m" :label="m" :value="m" />
                  </el-select>
                </el-form-item>
              </el-col>
            </el-row>

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

            <el-form-item label="上传图片">
              <el-upload
                v-if="detectionMode === 'single'"
                drag
                :limit="1"
                :auto-upload="false"
                :file-list="fileList"
                :on-change="handleFileChange"
                :on-remove="handleRemove"
                accept="image/*"
                class="compact-upload"
                style="width: 100%"
              >
                <el-icon class="upload-icon"><Upload /></el-icon>
                <div class="upload-text">拖拽或点击上传 · JPG/PNG</div>
              </el-upload>

              <el-upload
                v-else
                drag
                multiple
                :auto-upload="false"
                :file-list="fileList"
                :on-change="handleBatchChange"
                :on-remove="handleRemove"
                accept="image/*"
                class="compact-upload"
                style="width: 100%"
              >
                <el-icon class="upload-icon"><Upload /></el-icon>
                <div class="upload-text">拖拽或点击批量上传 · 最多50张</div>
              </el-upload>
            </el-form-item>

            <el-form-item>
              <el-button type="primary" :loading="loading" @click="handleDetect" style="width: 100%">
                开始检测
              </el-button>
              <el-button @click="handleReset" style="width: 100%; margin-top: 12px">
                重置
              </el-button>
            </el-form-item>
          </el-form>
        </el-card>
      </el-col>

      <!-- 右侧结果区 -->
      <el-col :xs="24" :lg="16">
        <el-card shadow="hover" class="result-card">
          <template #header>
            <span>检测结果</span>
          </template>

          <el-empty v-if="!singleResult && !batchResult" description="暂无检测结果" />

          <!-- 单张检测结果 -->
          <div v-if="singleResult" class="single-result">
            <el-alert
              :title="`严重程度: ${singleResult.severity}（指数 ${singleResult.severity_score.toFixed(1)}）`"
              :type="getSeverityType(singleResult.severity)"
              show-icon
              :closable="false"
              style="margin-bottom: 16px"
            >
              检测到 {{ singleResult.total_damages }} 处病害，耗时 {{ singleResult.processing_time.toFixed(2) }}s
            </el-alert>

            <!-- 原图 / 检测结果对比 -->
            <el-row :gutter="16" class="image-compare">
              <el-col :span="12">
                <div class="image-label">原始图像</div>
                <el-image
                  v-if="originalPreview"
                  :src="originalPreview"
                  fit="contain"
                  class="result-img"
                  :preview-src-list="[originalPreview]"
                  :initial-index="0"
                  preview-teleported
                />
              </el-col>
              <el-col :span="12">
                <div class="image-label">检测结果</div>
                <el-image
                  v-if="singleResult.result_image"
                  :src="singleResult.result_image"
                  fit="contain"
                  class="result-img"
                  :preview-src-list="[singleResult.result_image]"
                  :initial-index="0"
                  preview-teleported
                />
              </el-col>
            </el-row>

            <el-descriptions :column="2" border>
              <el-descriptions-item label="图片名称">{{ singleResult.image_name }}</el-descriptions-item>
              <el-descriptions-item label="检测框数">{{ singleResult.boxes.length }}</el-descriptions-item>
              <el-descriptions-item label="病害总数">{{ singleResult.total_damages }}</el-descriptions-item>
              <el-descriptions-item label="处理时间">{{ singleResult.processing_time.toFixed(2) }}s</el-descriptions-item>
            </el-descriptions>

            <template v-if="Object.keys(singleResult.class_counts).length">
              <el-divider />
              <h3>病害统计</h3>
              <el-row :gutter="16">
                <el-col v-for="(count, type) in singleResult.class_counts" :key="type" :span="8">
                  <el-statistic :title="String(type)" :value="count" />
                </el-col>
              </el-row>
            </template>
          </div>

          <!-- 批量检测结果 -->
          <div v-if="batchResult" class="batch-result">
            <el-alert
              :title="`总体严重程度: ${batchResult.overall_severity}（指数 ${batchResult.overall_severity_score.toFixed(1)}）`"
              :type="getSeverityType(batchResult.overall_severity)"
              show-icon
              :closable="false"
              style="margin-bottom: 16px"
            >
              共检测 {{ batchResult.total_images }} 张图片，发现 {{ batchResult.total_damages }} 处病害
            </el-alert>

            <el-descriptions :column="2" border style="margin-bottom: 16px">
              <el-descriptions-item label="检测图片数">{{ batchResult.total_images }}</el-descriptions-item>
              <el-descriptions-item label="病害总数">{{ batchResult.total_damages }}</el-descriptions-item>
            </el-descriptions>

            <template v-if="Object.keys(batchResult.class_counts).length">
              <h3>总体病害统计</h3>
              <el-row :gutter="16" style="margin-bottom: 24px">
                <el-col v-for="(count, type) in batchResult.class_counts" :key="type" :span="8">
                  <el-statistic :title="String(type)" :value="count" />
                </el-col>
              </el-row>
            </template>

            <h3>各图片详情</h3>
            <el-collapse>
              <el-collapse-item
                v-for="(result, index) in batchResult.results"
                :key="index"
                :title="`${result.image_name} - ${result.severity}`"
              >
                <el-descriptions :column="2" border>
                  <el-descriptions-item label="病害数量">{{ result.total_damages }}</el-descriptions-item>
                  <el-descriptions-item label="严重程度">
                    <el-tag :type="getSeverityType(result.severity)">{{ result.severity }}</el-tag>
                  </el-descriptions-item>
                  <el-descriptions-item label="处理时间">{{ result.processing_time.toFixed(2) }}s</el-descriptions-item>
                  <el-descriptions-item label="检测框数">{{ result.boxes.length }}</el-descriptions-item>
                </el-descriptions>
                <el-image
                  v-if="result.result_image"
                  :src="result.result_image"
                  fit="contain"
                  class="batch-result-img"
                  :preview-src-list="[result.result_image]"
                  :initial-index="0"
                  preview-teleported
                />
              </el-collapse-item>
            </el-collapse>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.detection-page {
  padding: 0;
}

.page-header {
  margin-bottom: 24px;
}

.page-header h1 {
  margin: 0 0 8px;
  font-size: 28px;
}

.page-header p {
  margin: 0;
}

.config-card,
.result-card {
  border-radius: 12px;
}

/* 紧凑表单：缩小项间距，减少整体高度 */
.compact-form :deep(.el-form-item) {
  margin-bottom: 10px;
}

.compact-form :deep(.el-form-item__label) {
  padding-bottom: 2px;
  line-height: 1.4;
}

.upload-icon {
  font-size: 36px;
  color: #409eff;
  margin-bottom: 8px;
}

.upload-text {
  font-size: 13px;
  color: #606266;
}

/* 压缩上传拖拽区高度 */
.compact-upload :deep(.el-upload-dragger) {
  padding: 16px 10px;
}

.single-result h3,
.batch-result h3 {
  margin: 24px 0 16px;
  font-size: 18px;
  color: #303133;
}

.image-compare {
  margin-bottom: 24px;
}

.image-label {
  margin-bottom: 8px;
  font-size: 14px;
  font-weight: 600;
  color: #606266;
}

.result-img {
  width: 100%;
  height: 280px;
  border-radius: 8px;
  border: 1px solid #ebeef5;
  background: #f5f7fa;
}

.batch-result-img {
  width: 100%;
  max-height: 360px;
  margin-top: 16px;
  border-radius: 8px;
  border: 1px solid #ebeef5;
  background: #f5f7fa;
}

@media (max-width: 992px) {
  .config-card {
    margin-bottom: 24px;
  }
}
</style>
