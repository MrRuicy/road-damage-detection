import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    redirect: '/dashboard'
  },
  {
    path: '/dashboard',
    name: 'Dashboard',
    component: () => import('../views/Dashboard.vue'),
    meta: { title: '仪表板' }
  },
  {
    path: '/detection',
    name: 'Detection',
    component: () => import('../views/Detection.vue'),
    meta: { title: '图像检测' }
  },
  {
    path: '/video',
    name: 'Video',
    component: () => import('../views/VideoDetection.vue'),
    meta: { title: '视频检测' }
  },
  {
    path: '/realtime',
    name: 'Realtime',
    component: () => import('../views/RealtimeDetection.vue'),
    meta: { title: '实时检测' }
  },
  {
    path: '/history',
    name: 'History',
    component: () => import('../views/History.vue'),
    meta: { title: '检测历史' }
  },
  {
    path: '/settings',
    name: 'Settings',
    component: () => import('../views/Settings.vue'),
    meta: { title: '系统设置' }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach((to, _from, next) => {
  document.title = `${to.meta.title} - 道路病害检测系统` || '道路病害检测系统'
  next()
})

export default router
