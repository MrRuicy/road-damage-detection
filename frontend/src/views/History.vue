<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  listRecords,
  deleteRecord,
  type RecordItem
} from '../api/records'
import { exportPdf, exportExcel } from '../api/export'

const records = ref<RecordItem[]>([])
const total = ref(0)
const loading = ref(false)

const query = reactive({
  page: 1,
  page_size: 10,
  detection_type: '',
  severity: ''
})

const detailVisible = ref(false)
const current = ref<RecordItem | null>(null)

const TYPE_LABELS: Record<string, string> = {
  image: '图像',
  batch: '批量',
  video: '视频',
  realtime: '实时'
}

const getSeverityType = (severity: string) => {
  const types: Record<string, any> = {
    无病害: 'success',
    轻微: 'success',
    中等: 'warning',
    严重: 'danger',
    危险: 'danger'
  }
  return types[severity] || 'info'
}

const load = async () => {
  loading.value = true
  try {
    const params: any = {
      page: query.page,
      page_size: query.page_size
    }
    if (query.detection_type) params.detection_type = query.detection_type
    if (query.severity) params.severity = query.severity

    const res = await listRecords(params)
    records.value = res.items
    total.value = res.total
  } catch (e: any) {
    ElMessage.error(e.message || '加载失败')
  } finally {
    loading.value = false
  }
}

const handleFilterChange = () => {
  query.page = 1
  load()
}

const handleReset = () => {
  query.detection_type = ''
  query.severity = ''
  query.page = 1
  load()
}

const showDetail = (row: RecordItem) => {
  current.value = row
  detailVisible.value = true
}

const handleDelete = async (row: RecordItem) => {
  try {
    await ElMessageBox.confirm(`确定删除记录「${row.source_name}」吗？`, '提示', {
      type: 'warning'
    })
    await deleteRecord(row.id)
    ElMessage.success('已删除')
    load()
  } catch (e: any) {
    if (e !== 'cancel') ElMessage.error(e.message || '删除失败')
  }
}

const formatTime = (t?: string) => {
  if (!t) return '-'
  return t.replace('T', ' ').slice(0, 19)
}

const exporting = ref(false)

const handleExportExcel = async () => {
  exporting.value = true
  try {
    const params: Record<string, any> = {}
    if (query.detection_type) params.detection_type = query.detection_type
    if (query.severity) params.severity = query.severity
    await exportExcel(params)
    ElMessage.success('Excel 导出成功')
  } catch (e: any) {
    ElMessage.error(e.message || '导出失败')
  } finally {
    exporting.value = false
  }
}

const handleExportPdf = async (row: RecordItem) => {
  try {
    await exportPdf(row.id)
    ElMessage.success('PDF 报告已下载')
  } catch (e: any) {
    ElMessage.error(e.message || '导出失败')
  }
}

onMounted(load)
</script>

<template>
  <div class="page">
    <div class="page-header">
      <h1>检测历史</h1>
      <p>查看与管理历史检测记录</p>
    </div>

    <el-card shadow="hover" class="filter-card">
      <div class="filter-bar">
        <el-select v-model="query.detection_type" placeholder="检测类型" clearable
          @change="handleFilterChange" style="width: 140px">
          <el-option label="图像" value="image" />
          <el-option label="批量" value="batch" />
          <el-option label="视频" value="video" />
        </el-select>
        <el-select v-model="query.severity" placeholder="严重程度" clearable
          @change="handleFilterChange" style="width: 140px">
          <el-option label="无病害" value="无病害" />
          <el-option label="轻微" value="轻微" />
          <el-option label="中等" value="中等" />
          <el-option label="严重" value="严重" />
          <el-option label="危险" value="危险" />
        </el-select>
        <el-button @click="handleReset">重置</el-button>
        <el-button type="primary" @click="load">刷新</el-button>
        <el-button type="success" :loading="exporting" @click="handleExportExcel">
          <el-icon><Download /></el-icon>导出 Excel
        </el-button>
      </div>
    </el-card>

    <el-card shadow="hover" class="table-card">
      <el-table :data="records" v-loading="loading" stripe>
        <el-table-column prop="id" label="ID" width="70" />
        <el-table-column prop="source_name" label="文件名" min-width="160" show-overflow-tooltip />
        <el-table-column label="类型" width="90">
          <template #default="{ row }">
            <el-tag size="small" effect="plain">{{ TYPE_LABELS[row.detection_type] || row.detection_type }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="total_damages" label="病害数" width="90" align="center" />
        <el-table-column label="严重程度" width="110" align="center">
          <template #default="{ row }">
            <el-tag :type="getSeverityType(row.severity)">{{ row.severity }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="severity_score" label="严重指数" width="100" align="center" />
        <el-table-column prop="model_name" label="模型" width="110" />
        <el-table-column label="检测时间" width="170">
          <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
        </el-table-column>
        <el-table-column label="操作" width="190" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" @click="showDetail(row)">详情</el-button>
            <el-button link type="success" @click="handleExportPdf(row)">PDF</el-button>
            <el-button link type="danger" @click="handleDelete(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>

      <div class="pagination">
        <el-pagination
          v-model:current-page="query.page"
          v-model:page-size="query.page_size"
          :total="total"
          :page-sizes="[10, 20, 50]"
          layout="total, sizes, prev, pager, next"
          @current-change="load"
          @size-change="handleFilterChange"
        />
      </div>
    </el-card>

    <!-- 详情抽屉 -->
    <el-drawer v-model="detailVisible" title="检测详情" size="480px">
      <div v-if="current" class="detail">
        <el-image
          v-if="current.result_image_path"
          :src="current.result_image_path"
          fit="contain"
          class="detail-img"
          :preview-src-list="[current.result_image_path]"
          preview-teleported
        />
        <el-descriptions :column="1" border>
          <el-descriptions-item label="文件名">{{ current.source_name }}</el-descriptions-item>
          <el-descriptions-item label="检测类型">{{ TYPE_LABELS[current.detection_type] || current.detection_type }}</el-descriptions-item>
          <el-descriptions-item label="模型">{{ current.model_name }}</el-descriptions-item>
          <el-descriptions-item label="置信度阈值">{{ current.conf_threshold }}</el-descriptions-item>
          <el-descriptions-item label="IOU阈值">{{ current.iou_threshold }}</el-descriptions-item>
          <el-descriptions-item label="病害总数">{{ current.total_damages }}</el-descriptions-item>
          <el-descriptions-item label="严重程度">
            <el-tag :type="getSeverityType(current.severity)">{{ current.severity }}</el-tag>
            <span style="margin-left: 8px">指数 {{ current.severity_score }}</span>
          </el-descriptions-item>
          <el-descriptions-item label="处理耗时">{{ current.processing_time.toFixed(2) }}s</el-descriptions-item>
          <el-descriptions-item label="检测时间">{{ formatTime(current.created_at) }}</el-descriptions-item>
        </el-descriptions>

        <template v-if="Object.keys(current.class_counts).length">
          <h4>病害明细</h4>
          <el-row :gutter="12">
            <el-col v-for="(count, type) in current.class_counts" :key="type" :span="8">
              <el-statistic :title="String(type)" :value="count" />
            </el-col>
          </el-row>
        </template>

        <el-button type="success" style="margin-top: 24px; width: 100%"
          @click="handleExportPdf(current)">
          <el-icon><Download /></el-icon>下载 PDF 报告
        </el-button>
      </div>
    </el-drawer>
  </div>
</template>

<style scoped>
.page {
  padding: 0;
}

.page-header {
  margin-bottom: 24px;
}

.page-header h1 {
  margin: 0 0 8px;
  font-size: 28px;
  color: #303133;
}

.page-header p {
  margin: 0;
  color: #909399;
}

.filter-card {
  border-radius: 12px;
  margin-bottom: 16px;
}

.filter-bar {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.table-card {
  border-radius: 12px;
}

.pagination {
  margin-top: 16px;
  display: flex;
  justify-content: flex-end;
}

.detail-img {
  width: 100%;
  height: 240px;
  border-radius: 8px;
  border: 1px solid #ebeef5;
  background: #f5f7fa;
  margin-bottom: 16px;
}

.detail h4 {
  margin: 20px 0 12px;
  color: #303133;
}
</style>
