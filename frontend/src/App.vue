<script setup lang="ts">
import { computed, ref, onMounted, onUnmounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useThemeStore } from './stores/theme'

const router = useRouter()
const route = useRoute()
const themeStore = useThemeStore()

// 高亮跟随当前路由，无论从菜单还是快捷操作跳转都同步
const activeMenu = computed(() => route.path)

// 移动端适配：侧边栏折叠状态
const isCollapsed = ref(false)
const isMobile = ref(false)

const checkScreenSize = () => {
  isMobile.value = window.innerWidth < 768
  // 移动端默认收起侧边栏
  if (isMobile.value && !isCollapsed.value) {
    isCollapsed.value = true
  }
}

onMounted(() => {
  checkScreenSize()
  window.addEventListener('resize', checkScreenSize)
})

onUnmounted(() => {
  window.removeEventListener('resize', checkScreenSize)
})

const toggleSidebar = () => {
  isCollapsed.value = !isCollapsed.value
}

const handleMenuSelect = (index: string) => {
  router.push(index)
  // 移动端点击菜单后自动收起侧边栏
  if (isMobile.value) {
    isCollapsed.value = true
  }
}
</script>

<template>
  <el-container class="app-container">
    <!-- 移动端汉堡菜单按钮（固定在右上角） -->
    <el-button
      v-if="isMobile"
      class="mobile-menu-btn"
      circle
      @click="toggleSidebar"
    >
      <el-icon><component :is="isCollapsed ? 'Menu' : 'Close'" /></el-icon>
    </el-button>

    <!-- 移动端遮罩层（侧边栏展开时显示） -->
    <transition name="fade">
      <div
        v-if="isMobile && !isCollapsed"
        class="sidebar-overlay"
        @click="toggleSidebar"
      />
    </transition>

    <el-aside
      :width="isCollapsed ? '0' : '240px'"
      class="app-aside"
      :class="{ 'is-collapsed': isCollapsed, 'is-mobile': isMobile }"
    >
      <div class="logo">
        <el-icon :size="32"><Position /></el-icon>
        <span class="logo-text">道路病害检测</span>
      </div>
      <el-menu
        :default-active="activeMenu"
        class="app-menu"
        @select="handleMenuSelect"
      >
        <el-menu-item index="/dashboard">
          <el-icon><Odometer /></el-icon>
          <span>仪表板</span>
        </el-menu-item>
        <el-menu-item index="/detection">
          <el-icon><Picture /></el-icon>
          <span>图像检测</span>
        </el-menu-item>
        <el-menu-item index="/video">
          <el-icon><VideoCamera /></el-icon>
          <span>视频检测</span>
        </el-menu-item>
        <el-menu-item index="/realtime">
          <el-icon><View /></el-icon>
          <span>实时检测</span>
        </el-menu-item>
        <el-menu-item index="/history">
          <el-icon><Clock /></el-icon>
          <span>检测历史</span>
        </el-menu-item>
        <el-menu-item index="/settings">
          <el-icon><Setting /></el-icon>
          <span>系统设置</span>
        </el-menu-item>
      </el-menu>

      <div class="aside-footer">
        <el-button text class="theme-toggle" @click="themeStore.toggle">
          <el-icon :size="18">
            <Moon v-if="!themeStore.isDark" />
            <Sunny v-else />
          </el-icon>
          <span>{{ themeStore.isDark ? '浅色模式' : '深色模式' }}</span>
        </el-button>
      </div>
    </el-aside>

    <el-container class="main-container">
      <el-main class="app-main">
        <router-view v-slot="{ Component }">
          <transition name="fade" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </el-main>
    </el-container>
  </el-container>
</template>

<style scoped>
.app-container {
  height: 100vh;
  position: relative;
}

/* 移动端汉堡菜单按钮 */
.mobile-menu-btn {
  position: fixed;
  top: 16px;
  right: 16px;
  z-index: 2001;
  background: var(--el-color-primary);
  color: #fff;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.15);
}

.mobile-menu-btn:hover {
  background: var(--el-color-primary-light-3);
}

/* 移动端遮罩层 */
.sidebar-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.5);
  z-index: 1999;
}

.app-aside {
  background: #001529;
  box-shadow: 2px 0 8px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  transition: width 0.3s ease, transform 0.3s ease;
  overflow: hidden;
}

/* 移动端侧边栏：固定定位，覆盖式 */
.app-aside.is-mobile {
  position: fixed;
  top: 0;
  left: 0;
  height: 100vh;
  z-index: 2000;
  width: 240px !important;
  transform: translateX(0);
}

.app-aside.is-mobile.is-collapsed {
  transform: translateX(-240px);
}

.logo {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 64px;
  color: #fff;
  font-size: 18px;
  font-weight: bold;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  gap: 12px;
  flex-shrink: 0;
}

.logo-text {
  background: linear-gradient(120deg, #4facfe 0%, #00f2fe 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.app-menu {
  border: none;
  background: #001529;
  flex: 1;
  overflow-y: auto;
}

.app-menu :deep(.el-menu-item) {
  color: rgba(255, 255, 255, 0.65);
}

.app-menu :deep(.el-menu-item:hover) {
  color: #fff;
  background: rgba(255, 255, 255, 0.08);
}

.app-menu :deep(.el-menu-item.is-active) {
  color: #fff;
  background: linear-gradient(120deg, #4facfe 0%, #00f2fe 100%);
}

.aside-footer {
  padding: 16px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  flex-shrink: 0;
}

.theme-toggle {
  width: 100%;
  color: rgba(255, 255, 255, 0.65);
  justify-content: flex-start;
  gap: 8px;
}

.theme-toggle:hover {
  color: #fff;
  background: rgba(255, 255, 255, 0.08);
}

.main-container {
  background: var(--el-bg-color-page);
}

.app-main {
  padding: 24px;
  overflow-y: auto;
}

/* 移动端：主区域占满全屏 */
@media (max-width: 767px) {
  .app-main {
    padding: 16px;
  }
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
