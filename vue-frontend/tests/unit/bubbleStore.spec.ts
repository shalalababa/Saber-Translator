import { beforeEach, describe, expect, it } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { createBubbleState } from '@/utils/bubbleFactory'

describe('bubbleStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('recomputes autoTextDirection when bubble coords change', () => {
    const bubbleStore = useBubbleStore()

    bubbleStore.setBubbles([
      createBubbleState({
        coords: [0, 0, 200, 100],
        polygon: [],
        textDirection: 'auto',
        autoTextDirection: 'horizontal',
      }),
    ])

    bubbleStore.updateBubble(0, {
      coords: [0, 0, 100, 220],
    })

    expect(bubbleStore.bubbles[0]?.autoTextDirection).toBe('vertical')
  })

  it('undoes and redoes deleting selected bubbles', () => {
    const bubbleStore = useBubbleStore()
    const first = createBubbleState({
      coords: [0, 0, 100, 60],
      polygon: [],
      translatedText: 'first',
    })
    const second = createBubbleState({
      coords: [120, 0, 220, 60],
      polygon: [],
      translatedText: 'second',
    })

    bubbleStore.setBubbles([first, second])
    bubbleStore.selectBubble(0)

    bubbleStore.deleteSelected()

    expect(bubbleStore.bubbles.map((bubble) => bubble.translatedText)).toEqual(['second'])
    expect(bubbleStore.canUndo).toBe(true)
    expect(bubbleStore.canRedo).toBe(false)

    expect(bubbleStore.undo()).toBe(true)

    expect(bubbleStore.bubbles.map((bubble) => bubble.translatedText)).toEqual(['first', 'second'])
    expect(bubbleStore.selectedIndex).toBe(0)
    expect(bubbleStore.canUndo).toBe(false)
    expect(bubbleStore.canRedo).toBe(true)

    expect(bubbleStore.redo()).toBe(true)

    expect(bubbleStore.bubbles.map((bubble) => bubble.translatedText)).toEqual(['second'])
    expect(bubbleStore.selectedIndex).toBe(-1)
  })

  it('undoes and redoes adding a bubble', () => {
    const bubbleStore = useBubbleStore()

    bubbleStore.setBubbles([])
    bubbleStore.addBubble([10, 20, 80, 120], {
      translatedText: 'new bubble',
    })

    expect(bubbleStore.bubbles).toHaveLength(1)
    expect(bubbleStore.bubbles[0]?.translatedText).toBe('new bubble')

    expect(bubbleStore.undo()).toBe(true)
    expect(bubbleStore.bubbles).toHaveLength(0)

    expect(bubbleStore.redo()).toBe(true)
    expect(bubbleStore.bubbles).toHaveLength(1)
    expect(bubbleStore.bubbles[0]?.translatedText).toBe('new bubble')
  })

  it('undoes and redoes bubble property edits', () => {
    const bubbleStore = useBubbleStore()

    bubbleStore.setBubbles([
      createBubbleState({
        coords: [0, 0, 100, 60],
        polygon: [],
        translatedText: 'before',
        fontSize: 24,
      }),
    ])

    bubbleStore.updateBubble(0, {
      translatedText: 'after',
      fontSize: 36,
    })

    expect(bubbleStore.bubbles[0]?.translatedText).toBe('after')
    expect(bubbleStore.bubbles[0]?.fontSize).toBe(36)

    expect(bubbleStore.undo()).toBe(true)
    expect(bubbleStore.bubbles[0]?.translatedText).toBe('before')
    expect(bubbleStore.bubbles[0]?.fontSize).toBe(24)

    expect(bubbleStore.redo()).toBe(true)
    expect(bubbleStore.bubbles[0]?.translatedText).toBe('after')
    expect(bubbleStore.bubbles[0]?.fontSize).toBe(36)
  })

  it('syncs undo and redo snapshots to the current image state', () => {
    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()

    imageStore.setImages([
      {
        id: 'img-1',
        fileName: 'test.png',
        originalDataURL: 'data:image/png;base64,abc',
        translatedDataURL: null,
        cleanImageData: null,
        bubbleStates: null,
        translationStatus: 'pending',
        translationFailed: false,
        hasUnsavedChanges: false,
      } as any,
    ])
    imageStore.setCurrentImageIndex(0)

    bubbleStore.setBubbles([
      createBubbleState({
        coords: [0, 0, 100, 60],
        polygon: [],
        translatedText: 'before',
      }),
    ])

    bubbleStore.updateBubble(0, { translatedText: 'after' })
    expect(imageStore.currentImage?.bubbleStates?.[0]?.translatedText).toBe('after')

    bubbleStore.undo()
    expect(imageStore.currentImage?.bubbleStates?.[0]?.translatedText).toBe('before')

    bubbleStore.redo()
    expect(imageStore.currentImage?.bubbleStates?.[0]?.translatedText).toBe('after')
  })
})
