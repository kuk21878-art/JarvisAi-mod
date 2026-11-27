#pragma once

// J.A.R.V.I.S. HUD - a standalone LeviLauncher native mod module.
//
// Three HUD panels in the in-game Mod Menu (they show up in the "External" tab):
//   Jarvis Status  - Mark II style panel: altitude, speed, heading, threat level
//   Jarvis Radar   - circular radar with hostile / passive / player contacts
//   Jarvis Scanner - block / entity under the crosshair + held item analysis
//
// Game data comes from signatures scanned in libminecraftpe.so and one hook on
// ClientInstance::update (both through the preloader's pl::memory), so no other mod
// has to be installed.
//
// Usage from your mod class (see MyMod.cpp):
//     #include "Jarvis.hpp"
//     enable():  jarvis::start(getSelf().getId(), getSelf().getConfigDir());
//     disable(): jarvis::stop();

#include "JarvisGame.hpp"

#include <jni.h>
#include <pl/ModMenu.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <map>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <fcntl.h>
#include <set>
#include <unistd.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <vector>

namespace jarvis {

// ===========================================================================
// Shared data (filled on the local player tick, read when drawing)
// ===========================================================================

struct Contact {
    float dx = 0.f, dy = 0.f, dz = 0.f;   // world offset from the player (blocks)
    float dist = 0.f;
    unsigned cats = 0;                    // raw Actor category bits (for calibration)
    bool player = false;
    bool hostile = false;
    std::string type;                     // "cow", "zombie" ... (empty when unknown)
    float wx = 0.f, wy = 0.f, wz = 0.f;   // world position of the mob's feet (for the glow boxes)
    float width = 0.6f, height = 1.8f;    // approximate hitbox size
};

struct Target {
    enum class Kind { None, Block, Entity };
    Kind kind = Kind::None;
    std::string name;
    float dist = 0.f;
    bool hostile = false;
    bool player = false;
    unsigned cats = 0;
    std::string type;      // entity id without namespace, empty when unknown
    bool hasBox = false;   // world-space bounding box of the target (for the AR overlay)
    float minX = 0.f, minY = 0.f, minZ = 0.f;
    float maxX = 0.f, maxY = 0.f, maxZ = 0.f;
};

struct HeldItem {
    bool valid = false;
    std::string name;
    int damage = 0;
    int maxDamage = 0;
};

struct Snapshot {
    bool valid = false;                   // false when the player is not ticking (menu / loading)
    float x = 0.f, y = 0.f, z = 0.f;
    float yaw = 0.f, pitch = 0.f;
    float speed = 0.f;                    // horizontal blocks per second
    std::vector<Contact> contacts;        // nearest first
    int hostileCount = 0;
    float nearestHostile = -1.f;
    Target target;
    HeldItem held;
};

struct Settings {
    std::atomic<bool> needRadar{false};
    std::atomic<bool> needScanner{false};
    std::atomic<float> radarRange{32.f};
    std::atomic<int> hostileMask{4};      // Actor category bit(s) treated as "hostile"
    std::atomic<bool> showPlayers{true};
    std::atomic<bool> showPassive{true};
    std::atomic<float> entityHeight{1.8f};   // assumed mob height for the AR box
    std::atomic<bool> debug{false};          // show calibration info (radar "Debug Categories")
    std::atomic<bool> aiNeeds{false};        // the AI panel wants radar + target data
    std::atomic<bool> needGlow{false};       // the Spectral Glow module wants every mob around
    std::atomic<float> glowRange{40.f};
    std::atomic<float> arFov{70.f};          // vertical field of view shared by the AR overlays
};

namespace detail {

// Same layout BedrockTools' Hitbox module uses for ActorFetchNearbyActorsSorted.
struct ActorEntry {
    void* actor;
    float distance;
    float pad;
};

struct ActorVec {   // == std::vector<ActorEntry> (begin / end / capacity)
    ActorEntry* begin;
    ActorEntry* end;
    ActorEntry* cap;
};

using FetchNearbyFn = ActorVec (*)(void* actor, void* extent, int actorType);
using IsPlayerFn = bool (*)(void* actor);
using GetHitResultFn = void* (*)(void* level);
using HitGetEntityFn = void* (*)(void* hitResult);
using GetBlockFn = void* (*)(void* blockSource, const jsdk::BlockPos& pos);
using RawNameFn = std::string (*)(void* itemStack);
using DamageFn = short (*)(void* itemStack);
using LocalPlayerFn = void* (*)(void* clientInstance);
using UpdateFn = void* (*)(void* clientInstance, bool value);

inline constexpr int kMobActorType = 0x100;      // same filter value the Hitbox module passes
inline constexpr unsigned kCategoryMob = 2;      // Actor category bit "mob"
inline constexpr int kScanIntervalMs = 200;      // radar refresh: 5 Hz
inline constexpr int kTargetIntervalMs = 100;    // scanner refresh: 10 Hz
inline constexpr std::size_t kMaxContacts = 64;
inline constexpr int kHitTile = 0;               // HitResultType::Tile
inline constexpr int kHitEntity = 1;             // HitResultType::Entity

class Panel;

struct Core {
    std::mutex mutex;
    Snapshot snap;
    Settings settings;
    std::chrono::steady_clock::time_point lastTick{};

    // game functions (resolved from signatures)
    FetchNearbyFn fetchNearby = nullptr;
    IsPlayerFn isPlayer = nullptr;
    GetHitResultFn getHitResult = nullptr;
    HitGetEntityFn hitGetEntity = nullptr;
    GetBlockFn getBlock = nullptr;
    RawNameFn rawName = nullptr;
    DamageFn damageValue = nullptr;

    LocalPlayerFn getLocalPlayer = nullptr;

    bool havePrev = false;
    jsdk::Vec3 prev{};
    float speed = 0.f;
    std::chrono::steady_clock::time_point lastUpdate{};
    std::chrono::steady_clock::time_point lastRadar{};
    std::chrono::steady_clock::time_point lastTarget{};

    // game hook / lifecycle
    UpdateFn updateOriginal = nullptr;
    void* hookTarget = nullptr;
    bool hooked = false;
    bool resolved = false;
    std::atomic<bool> active{false};
    std::atomic<int> state{0};   // 0 waiting for game, 1 online, 2 unsupported game version, 3 hook failed
    std::thread attachThread;
    std::atomic<bool> stopping{false};
    bool started = false;
    bool registered = false;

    // config
    std::string modId;
    std::filesystem::path configPath;
    std::vector<std::shared_ptr<Panel>> panels;   // shared_ptr: Panel is incomplete at this point
};

inline Core& core() {
    static Core instance;
    return instance;
}

inline bool finite3(const jsdk::Vec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

inline std::string prettify(std::string text) {
    constexpr std::string_view prefix = "minecraft:";
    if (text.rfind(prefix, 0) == 0) text.erase(0, prefix.size());
    for (auto& ch : text) {
        ch = ch == '_' ? ' ' : static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return text;
}

inline void** vtableOf(void* object) {
    return object ? *reinterpret_cast<void***>(object) : nullptr;
}

// Copies the scanned addresses into typed function pointers (0 -> nullptr -> feature off).
inline void bindSignatures() {
    auto& c = core();
    using jsdk::Sig;
    auto bind = [](auto& target, Sig id) {
        target = reinterpret_cast<std::remove_reference_t<decltype(target)>>(jsdk::address(id));
    };
    bind(c.fetchNearby, Sig::ActorFetchNearbyActorsSorted);
    bind(c.isPlayer, Sig::ActorIsPlayer);
    bind(c.getHitResult, Sig::LevelGetHitResult);
    bind(c.hitGetEntity, Sig::HitResultGetEntity);
    bind(c.getBlock, Sig::BlockSourceGetBlock);
    bind(c.rawName, Sig::ItemStackBaseGetRawNameId);
    bind(c.damageValue, Sig::ItemStackBaseGetDamageValue);
    bind(c.getLocalPlayer, Sig::ClientInstanceGetLocalPlayer);
}

// ===========================================================================
// World scanning (runs on the local player tick)
// ===========================================================================

// ---------------------------------------------------------------------------
// Entity type names ("cow", "zombie"), discovered at runtime
// ---------------------------------------------------------------------------
//
// There is no signature for "what kind of mob is this", so the actor's memory is searched for
// an identifier string such as "minecraft:cow" (directly inside the actor or one pointer
// away). Once the same location has produced two different names on two different mobs it
// is trusted and reused, which makes later lookups cheap.

inline constexpr std::string_view kKnownMobs[] = {
    "allay", "armadillo", "axolotl", "bat", "bee", "blaze", "bogged", "breeze", "camel", "cat", "cave_spider",
    "chicken", "cod", "cow", "creaking", "creeper", "dolphin", "donkey", "drowned", "elder_guardian", "enderman",
    "endermite", "ender_dragon", "evoker", "fox", "frog", "ghast", "glow_squid", "goat", "guardian", "hoglin",
    "horse", "husk", "iron_golem", "llama", "magma_cube", "mooshroom", "mule", "ocelot", "panda", "parrot",
    "phantom", "pig", "piglin", "piglin_brute", "pillager", "polar_bear", "pufferfish", "rabbit", "ravager",
    "salmon", "sheep", "shulker", "silverfish", "skeleton", "skeleton_horse", "slime", "sniffer", "snow_golem",
    "spider", "squid", "stray", "strider", "tadpole", "trader_llama", "tropicalfish", "turtle", "vex",
    "villager", "villager_v2", "vindicator", "wandering_trader", "warden", "witch", "wither",
    "wither_skeleton", "wolf", "zoglin", "zombie", "zombie_horse", "zombie_villager", "zombie_villager_v2",
    "zombie_pigman", "zombified_piglin", "player"};

inline constexpr std::string_view kHostileMobs[] = {
    "blaze", "bogged", "breeze", "cave_spider", "creaking", "creeper", "drowned", "elder_guardian", "endermite",
    "ender_dragon", "evoker", "ghast", "guardian", "hoglin", "husk", "magma_cube", "phantom", "piglin_brute",
    "pillager", "ravager", "shulker", "silverfish", "skeleton", "slime", "spider", "stray", "vex", "vindicator",
    "warden", "witch", "wither", "wither_skeleton", "zoglin", "zombie", "zombie_villager", "zombie_villager_v2"};

struct MobSize { std::string_view name; float width; float height; };
inline constexpr MobSize kMobSizes[] = {
    {"cow", 0.9f, 1.4f}, {"mooshroom", 0.9f, 1.4f}, {"pig", 0.9f, 0.9f}, {"sheep", 0.9f, 1.3f},
    {"chicken", 0.4f, 0.7f}, {"zombie", 0.6f, 1.95f}, {"husk", 0.6f, 1.95f}, {"drowned", 0.6f, 1.95f},
    {"skeleton", 0.6f, 1.99f}, {"stray", 0.6f, 1.99f}, {"creeper", 0.6f, 1.7f}, {"spider", 1.4f, 0.9f},
    {"cave_spider", 0.7f, 0.5f}, {"enderman", 0.6f, 2.9f}, {"villager", 0.6f, 1.95f}, {"villager_v2", 0.6f, 1.95f},
    {"witch", 0.6f, 1.95f}, {"horse", 1.4f, 1.6f}, {"donkey", 1.4f, 1.6f}, {"wolf", 0.6f, 0.85f},
    {"cat", 0.6f, 0.7f}, {"rabbit", 0.4f, 0.5f}, {"bat", 0.5f, 0.9f}, {"iron_golem", 1.4f, 2.7f},
    {"blaze", 0.6f, 1.8f}, {"ghast", 4.0f, 4.0f}, {"phantom", 0.9f, 0.5f}, {"squid", 0.8f, 0.8f},
    {"slime", 0.52f, 0.52f}, {"silverfish", 0.4f, 0.3f}, {"fox", 0.6f, 0.7f}, {"player", 0.6f, 1.8f},
};

struct MobHealth { std::string_view name; float hp; };
// Vanilla MAXIMUM health. The current health of a mob cannot be read in this version.
inline constexpr MobHealth kMobHealth[] = {
    {"cow", 10}, {"mooshroom", 10}, {"pig", 10}, {"sheep", 8}, {"chicken", 4}, {"zombie", 20}, {"husk", 20},
    {"drowned", 20}, {"zombie_villager", 20}, {"zombie_villager_v2", 20}, {"zombie_pigman", 20},
    {"zombified_piglin", 20}, {"skeleton", 20}, {"stray", 20}, {"bogged", 16}, {"wither_skeleton", 20},
    {"creeper", 20}, {"spider", 16}, {"cave_spider", 12}, {"enderman", 40}, {"endermite", 8},
    {"villager", 20}, {"villager_v2", 20}, {"witch", 26}, {"horse", 30}, {"donkey", 30}, {"mule", 30},
    {"llama", 30}, {"wolf", 8}, {"cat", 10}, {"ocelot", 10}, {"rabbit", 3}, {"bat", 6}, {"iron_golem", 100},
    {"snow_golem", 4}, {"blaze", 20}, {"ghast", 10}, {"phantom", 20}, {"squid", 10}, {"glow_squid", 10},
    {"silverfish", 8}, {"fox", 10}, {"player", 20}, {"wither", 300}, {"ender_dragon", 200}, {"warden", 500},
    {"evoker", 24}, {"vindicator", 24}, {"pillager", 24}, {"ravager", 100}, {"shulker", 30}, {"guardian", 30},
    {"elder_guardian", 80}, {"hoglin", 40}, {"zoglin", 40}, {"piglin", 16}, {"piglin_brute", 50}, {"vex", 14},
    {"strider", 20}, {"turtle", 30}, {"panda", 20}, {"polar_bear", 30}, {"parrot", 6}, {"dolphin", 10},
    {"cod", 3}, {"salmon", 3}, {"pufferfish", 3}, {"tropicalfish", 3}, {"bee", 10}, {"goat", 10},
    {"axolotl", 14}, {"frog", 10}, {"tadpole", 6}, {"allay", 20}, {"camel", 32}, {"sniffer", 14},
    {"armadillo", 12}, {"breeze", 30}, {"trader_llama", 30}, {"wandering_trader", 20}, {"zombie_horse", 15},
    {"skeleton_horse", 15},
};

inline bool mobMaxHealth(std::string_view type, float& hp) {
    for (const auto& entry : kMobHealth) {
        if (entry.name == type) {
            hp = entry.hp;
            return true;
        }
    }
    return false;
}

template <std::size_t N>
inline bool inList(const std::string_view (&list)[N], std::string_view value) {
    for (const auto& entry : list) {
        if (entry == value) return true;
    }
    return false;
}

inline bool isHostileType(std::string_view type) { return inList(kHostileMobs, type); }

inline bool mobSize(std::string_view type, float& width, float& height) {
    for (const auto& entry : kMobSizes) {
        if (entry.name == type) {
            width = entry.width;
            height = entry.height;
            return true;
        }
    }
    return false;
}

// Reads memory without risking a crash: write() reports EFAULT instead of faulting.
inline std::size_t readMemory(const void* source, void* destination, std::size_t size) {
    static int fds[2] = {-1, -1};
    static bool broken = false;
    if (broken || !source || !destination || size == 0) return 0;
    if (fds[0] < 0 && pipe2(fds, O_NONBLOCK | O_CLOEXEC) != 0) {
        broken = true;
        return 0;
    }
    std::size_t total = 0;
    while (total < size) {
        const std::size_t chunk = std::min<std::size_t>(size - total, 1024);
        const ssize_t written = write(fds[1], static_cast<const char*>(source) + total, chunk);
        if (written <= 0) break;
        std::size_t drained = 0;
        while (drained < static_cast<std::size_t>(written)) {
            const ssize_t got = read(fds[0], static_cast<char*>(destination) + total + drained,
                                     static_cast<std::size_t>(written) - drained);
            if (got <= 0) return total + drained;
            drained += static_cast<std::size_t>(got);
        }
        total += drained;
        if (static_cast<std::size_t>(written) < chunk) break;
    }
    return total;
}

inline bool looksLikeIdentifier(const std::string& text) {
    if (text.size() < 5 || text.size() > 48) return false;
    std::size_t colon = std::string::npos;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char ch = text[i];
        if (ch == ':') {
            if (colon != std::string::npos || i == 0 || i + 1 == text.size()) return false;
            colon = i;
        } else if (!((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_' || ch == '.' || ch == '-')) {
            return false;
        }
    }
    return colon != std::string::npos;
}

// Decodes a libc++ std::string stored at `block` (24 bytes).
inline bool readStdString(const unsigned char* block, std::string& out) {
    const unsigned char first = block[0];
    if ((first & 1u) == 0) {   // short string, characters follow the size byte
        const std::size_t size = first >> 1;
        if (size < 5 || size > 22) return false;
        out.assign(reinterpret_cast<const char*>(block + 1), size);
        return true;
    }
    std::uint64_t size = 0, pointer = 0;
    std::memcpy(&size, block + 8, 8);
    std::memcpy(&pointer, block + 16, 8);
    if (size < 5 || size > 48 || pointer < 0x10000 || pointer > 0x0000FFFFFFFFFFFFull) return false;
    char buffer[64];
    if (readMemory(reinterpret_cast<const void*>(pointer), buffer, static_cast<std::size_t>(size)) != size) return false;
    out.assign(buffer, static_cast<std::size_t>(size));
    return true;
}

struct TypePath {
    int pointerOffset = -1;   // -1: the string lives inside the actor itself
    int stringOffset = 0;
};

struct TypeFinder {
    std::vector<TypePath> candidates;
    std::vector<std::set<std::string>> seen;
    std::vector<bool> knownHit;
    int confirmed = -1;
    int evaluations = 0;
    std::chrono::steady_clock::time_point lastDiscovery{};
    std::string debug;
};

inline TypeFinder& typeFinder() {
    static TypeFinder finder;
    return finder;
}

inline std::string stripNamespace(const std::string& id) {
    const auto colon = id.find(':');
    return colon == std::string::npos ? id : id.substr(colon + 1);
}

inline bool evalTypePath(const void* actor, const TypePath& path, std::string& out) {
    unsigned char block[24];
    if (path.pointerOffset < 0) {
        if (readMemory(static_cast<const char*>(actor) + path.stringOffset, block, 24) != 24) return false;
    } else {
        std::uint64_t pointer = 0;
        if (readMemory(static_cast<const char*>(actor) + path.pointerOffset, &pointer, 8) != 8) return false;
        if (pointer < 0x10000 || pointer > 0x0000FFFFFFFFFFFFull || (pointer & 7u) != 0) return false;
        if (readMemory(reinterpret_cast<const char*>(pointer) + path.stringOffset, block, 24) != 24) return false;
    }
    std::string value;
    if (!readStdString(block, value) || !looksLikeIdentifier(value)) return false;
    out = std::move(value);
    return true;
}

inline void discoverTypePaths(const void* actor) {
    auto& finder = typeFinder();
    struct Hit { TypePath path; std::string value; };
    std::vector<Hit> hits;

    auto scan = [&](const unsigned char* data, std::size_t length, int pointerOffset) {
        for (std::size_t off = 0; off + 24 <= length && hits.size() < 64; off += 8) {
            std::string value;
            if (readStdString(data + off, value) && looksLikeIdentifier(value)) {
                hits.push_back({{pointerOffset, static_cast<int>(off)}, value});
            }
        }
    };

    std::vector<unsigned char> body(0x800);
    const std::size_t bodySize = readMemory(actor, body.data(), body.size());
    scan(body.data(), bodySize, -1);

    std::vector<unsigned char> target(0x300);
    int dereferenced = 0;
    for (std::size_t off = 0; off + 8 <= std::min<std::size_t>(bodySize, 0x600) && dereferenced < 220; off += 8) {
        std::uint64_t pointer = 0;
        std::memcpy(&pointer, body.data() + off, 8);
        if (pointer < 0x10000 || pointer > 0x0000FFFFFFFFFFFFull || (pointer & 7u) != 0) continue;
        ++dereferenced;
        const std::size_t got = readMemory(reinterpret_cast<const void*>(pointer), target.data(), target.size());
        if (got >= 24) scan(target.data(), got, static_cast<int>(off));
    }

    // Prefer locations that produced a real mob name; otherwise anything in the minecraft namespace.
    std::vector<Hit> keep;
    for (const auto& hit : hits) {
        if (inList(kKnownMobs, stripNamespace(hit.value))) keep.push_back(hit);
    }
    if (keep.empty()) {
        for (const auto& hit : hits) {
            if (hit.value.rfind("minecraft:", 0) == 0) keep.push_back(hit);
        }
    }
    if (keep.size() > 24) keep.resize(24);

    finder.candidates.clear();
    finder.seen.clear();
    finder.knownHit.clear();
    finder.confirmed = -1;
    finder.evaluations = 0;
    for (const auto& hit : keep) {
        finder.candidates.push_back(hit.path);
        finder.seen.emplace_back();
        finder.knownHit.push_back(false);
    }
}

// Returns the entity id without namespace ("cow"), or an empty string when it is not known.
inline std::string entityTypeId(const void* actor) {
    auto& finder = typeFinder();
    if (!actor) return {};
    std::string value;

    if (finder.confirmed >= 0) {
        if (evalTypePath(actor, finder.candidates[static_cast<std::size_t>(finder.confirmed)], value)) {
            return stripNamespace(value);
        }
        return {};
    }

    if (finder.candidates.empty()) {
        const auto now = std::chrono::steady_clock::now();
        if (now - finder.lastDiscovery < std::chrono::seconds(1)) return {};
        finder.lastDiscovery = now;
        discoverTypePaths(actor);
        if (finder.candidates.empty()) return {};
    }

    ++finder.evaluations;
    std::string best;
    for (std::size_t i = 0; i < finder.candidates.size(); ++i) {
        if (!evalTypePath(actor, finder.candidates[i], value)) continue;
        const std::string name = stripNamespace(value);
        finder.seen[i].insert(name);
        const bool known = inList(kKnownMobs, name);
        if (known) finder.knownHit[i] = true;
        if (best.empty() || (known && !inList(kKnownMobs, best))) best = name;
        if (finder.seen[i].size() >= 2 && finder.knownHit[i]) {
            finder.confirmed = static_cast<int>(i);
            break;
        }
    }
    // Only one kind of mob around for a long time: settle on the first location that gave a real name.
    if (finder.confirmed < 0 && finder.evaluations > 80) {
        for (std::size_t i = 0; i < finder.candidates.size(); ++i) {
            if (finder.knownHit[i]) { finder.confirmed = static_cast<int>(i); break; }
        }
    }
    if (finder.confirmed >= 0) {
        const TypePath& path = finder.candidates[static_cast<std::size_t>(finder.confirmed)];
        char text[64];
        if (path.pointerOffset >= 0) {
            std::snprintf(text, sizeof(text), "ID: ptr +0x%X > +0x%X", path.pointerOffset, path.stringOffset);
        } else {
            std::snprintf(text, sizeof(text), "ID: direct +0x%X", path.stringOffset);
        }
        finder.debug = text;
        if (evalTypePath(actor, path, value)) return stripNamespace(value);
    }
    return best;
}

inline void scanContacts(jsdk::Player* player, const jsdk::Vec3& me, Snapshot& out) {
    auto& c = core();
    out.contacts.clear();
    out.hostileCount = 0;
    out.nearestHostile = -1.f;
    if (!c.fetchNearby) return;

    const bool glow = c.settings.needGlow.load();
    float range = std::clamp(c.settings.radarRange.load(), 8.f, 180.f);
    if (glow) range = std::max(range, std::clamp(c.settings.glowRange.load(), 8.f, 180.f));
    const unsigned mask = static_cast<unsigned>(c.settings.hostileMask.load());
    const bool showPlayers = c.settings.showPlayers.load();
    const bool showPassive = c.settings.showPassive.load();

    jsdk::Vec3 extent{range, range, range};
    ActorVec found = c.fetchNearby(player, &extent, kMobActorType);

    if (found.begin && found.end) {
        for (ActorEntry* it = found.begin; it < found.end && out.contacts.size() < kMaxContacts; ++it) {
            void* raw = it->actor;
            if (!raw || raw == static_cast<void*>(player)) continue;

            auto* actor = reinterpret_cast<jsdk::Actor*>(raw);
            const jsdk::Vec3 pos = actor->position();
            if (!finite3(pos)) continue;

            Contact contact;
            contact.dx = pos.x - me.x;
            contact.dy = pos.y - me.y;
            contact.dz = pos.z - me.z;
            contact.dist = std::sqrt(contact.dx * contact.dx + contact.dy * contact.dy + contact.dz * contact.dz);
            if (contact.dist > range * 1.75f) continue;

            contact.player = c.isPlayer ? c.isPlayer(raw) : false;
            contact.cats = actor->categories();
            if (!contact.player && (contact.cats & kCategoryMob) == 0) continue;   // items, arrows, xp ...
            if (!contact.player) contact.type = entityTypeId(raw);
            contact.hostile = !contact.player &&
                              (contact.type.empty() ? (contact.cats & mask) != 0 : isHostileType(contact.type));

            // The radar applies these two filters itself when drawing; the glow wants everything.
            if (!glow) {
                if (contact.player && !showPlayers) continue;
                if (!contact.player && !contact.hostile && !showPassive) continue;
            }

            contact.wx = pos.x;
            contact.wz = pos.z;
            contact.wy = contact.player ? pos.y - 1.62f : pos.y;   // mobs are stored at their feet
            contact.height = std::clamp(c.settings.entityHeight.load(), 0.3f, 6.f);
            if (contact.player) mobSize("player", contact.width, contact.height);
            else if (!contact.type.empty()) mobSize(contact.type, contact.width, contact.height);

            if (contact.hostile) {
                ++out.hostileCount;
                if (out.nearestHostile < 0.f || contact.dist < out.nearestHostile) out.nearestHostile = contact.dist;
            }
            out.contacts.push_back(contact);
        }
    }

    // The returned vector is owned by the caller (it is returned by value).
    if (found.begin) ::operator delete(found.begin);

    std::sort(out.contacts.begin(), out.contacts.end(),
              [](const Contact& a, const Contact& b) { return a.dist < b.dist; });
}

inline void* carriedItemStack(jsdk::Player* player) {
    void** vtable = vtableOf(player);
    const auto slot = jsdk::offsets::VTable::PlayerGetCarriedItem;
    if (!vtable || !vtable[slot]) return nullptr;
    return reinterpret_cast<void* (*)(void*)>(vtable[slot])(player);
}

inline int stackMaxDamage(void* stack) {
    if (!stack) return 0;
    namespace off = jsdk::offsets;
    void* counter = *reinterpret_cast<void**>(reinterpret_cast<std::byte*>(stack) + off::ItemStack::ItemStackBaseItem);
    if (!counter) return 0;
    void* item = *reinterpret_cast<void**>(reinterpret_cast<std::byte*>(counter) + off::ItemStack::SharedCounterPointer);
    void** vtable = vtableOf(item);
    if (!vtable || !vtable[off::VTable::ItemGetMaxDamage]) return 0;
    return reinterpret_cast<short (*)(void*)>(vtable[off::VTable::ItemGetMaxDamage])(item);
}

inline void scanHeldItem(jsdk::Player* player, HeldItem& out) {
    auto& c = core();
    out = HeldItem{};
    if (!c.rawName) return;
    void* stack = carriedItemStack(player);
    if (!stack) return;
    std::string name = c.rawName(stack);
    if (name.empty() || name == "air" || name.size() > 96) return;
    out.valid = true;
    out.name = prettify(name);
    out.damage = c.damageValue ? static_cast<int>(c.damageValue(stack)) : 0;
    out.maxDamage = stackMaxDamage(stack);
}

inline void scanTarget(jsdk::Player* player, Target& out) {
    auto& c = core();
    out = Target{};

    void* level = player->level();
    if (!level) return;

    void* hit = c.getHitResult ? c.getHitResult(level) : nullptr;
    if (!hit) hit = jsdk::storedHitResult(level);
    if (!hit) return;

    namespace off = jsdk::offsets;
    const int type = *reinterpret_cast<int*>(reinterpret_cast<std::byte*>(hit) + off::HitResult::mType);
    const jsdk::Vec3 start = *reinterpret_cast<jsdk::Vec3*>(reinterpret_cast<std::byte*>(hit) + off::HitResult::mStartPos);
    const jsdk::Vec3 hitPos = *reinterpret_cast<jsdk::Vec3*>(reinterpret_cast<std::byte*>(hit) + off::HitResult::mPos);
    if (!finite3(start) || !finite3(hitPos)) return;

    const float dx = hitPos.x - start.x;
    const float dy = hitPos.y - start.y;
    const float dz = hitPos.z - start.z;
    const float length = std::sqrt(dx * dx + dy * dy + dz * dz);

    if (type == kHitEntity && c.hitGetEntity) {
        void* raw = c.hitGetEntity(hit);
        if (!raw) return;
        auto* actor = reinterpret_cast<jsdk::Actor*>(raw);
        out.kind = Target::Kind::Entity;
        out.dist = length;
        out.cats = actor->categories();
        out.player = c.isPlayer ? c.isPlayer(raw) : false;
        if (!out.player) out.type = entityTypeId(raw);
        out.hostile = !out.player &&
                      (out.type.empty() ? (out.cats & static_cast<unsigned>(c.settings.hostileMask.load())) != 0
                                        : isHostileType(out.type));

        // A mob's stored position is at its feet (a player's is at eye level, 1.62 blocks up).
        const jsdk::Vec3 at = actor->position();
        if (finite3(at)) {
            float width = 0.6f;
            float height = std::clamp(c.settings.entityHeight.load(), 0.3f, 6.f);
            if (!out.type.empty()) mobSize(out.type, width, height);
            if (out.player) mobSize("player", width, height);
            const float feet = out.player ? at.y - 1.62f : at.y;
            out.hasBox = true;
            out.minX = at.x - width * 0.5f; out.maxX = at.x + width * 0.5f;
            out.minZ = at.z - width * 0.5f; out.maxZ = at.z + width * 0.5f;
            out.minY = feet;
            out.maxY = feet + height;
        }
        return;
    }

    if (type == kHitTile && c.getBlock && length > 1e-4f) {
        // Nudge the hit point slightly into the block so floor() lands on the hit block.
        const float nudge = 0.01f / length;
        const jsdk::BlockPos pos{
            static_cast<int>(std::floor(hitPos.x + dx * nudge)),
            static_cast<int>(std::floor(hitPos.y + dy * nudge)),
            static_cast<int>(std::floor(hitPos.z + dz * nudge))};

        void* region = jsdk::blockSourceOf(player->dimension());
        if (!region) return;

        void* block = c.getBlock(region, pos);
        const std::string* fullName = jsdk::blockFullName(block);
        if (!fullName || fullName->empty() || fullName->size() > 96) return;

        out.kind = Target::Kind::Block;
        out.name = prettify(*fullName);
        out.dist = length;
        out.hasBox = true;
        out.minX = static_cast<float>(pos.x); out.maxX = out.minX + 1.f;
        out.minY = static_cast<float>(pos.y); out.maxY = out.minY + 1.f;
        out.minZ = static_cast<float>(pos.z); out.maxZ = out.minZ + 1.f;
    }
}

inline void onTick(jsdk::Player* player) {
    if (!player) return;
    auto& c = core();

    const jsdk::Vec3 pos = player->position();
    if (!finite3(pos)) return;
    const jsdk::Vec2 rotation = player->rotation();

    const auto now = std::chrono::steady_clock::now();
    const float dt = c.lastUpdate.time_since_epoch().count() != 0
        ? std::chrono::duration<float>(now - c.lastUpdate).count() : 0.f;
    c.lastUpdate = now;

    if (c.havePrev && dt > 0.001f && dt < 0.5f) {
        const float dx = pos.x - c.prev.x;
        const float dz = pos.z - c.prev.z;
        float instant = std::sqrt(dx * dx + dz * dz) / dt;
        if (instant > 80.f) instant = 0.f;                      // teleport / dimension change
        c.speed += (instant - c.speed) * std::min(1.f, dt * 5.f);   // ~0.2 s smoothing
    }
    c.prev = pos;
    c.havePrev = true;

    auto due = [&](std::chrono::steady_clock::time_point& last, int intervalMs) {
        if (now - last < std::chrono::milliseconds(intervalMs)) return false;
        last = now;
        return true;
    };

    Snapshot scratch;
    const bool aiWants = c.settings.aiNeeds.load();
    const bool glowWants = c.settings.needGlow.load();
    const bool doRadar = (c.settings.needRadar.load() || aiWants || glowWants) &&
                         due(c.lastRadar, glowWants ? 50 : kScanIntervalMs);
    const bool doTarget = (c.settings.needScanner.load() || aiWants) && due(c.lastTarget, kTargetIntervalMs);
    if (doRadar) scanContacts(player, pos, scratch);
    if (doTarget) {
        scanTarget(player, scratch.target);
        scanHeldItem(player, scratch.held);
    }

    std::lock_guard<std::mutex> lock(c.mutex);
    c.lastTick = now;
    c.snap.x = pos.x;
    c.snap.y = pos.y;
    c.snap.z = pos.z;
    c.snap.pitch = rotation.x;
    c.snap.yaw = rotation.y;
    c.snap.speed = c.speed;
    if (doRadar) {
        c.snap.contacts = std::move(scratch.contacts);
        c.snap.hostileCount = scratch.hostileCount;
        c.snap.nearestHostile = scratch.nearestHostile;
    }
    if (doTarget) {
        c.snap.target = std::move(scratch.target);
        c.snap.held = std::move(scratch.held);
    }
}

inline Snapshot snapshot() {
    auto& c = core();
    std::lock_guard<std::mutex> lock(c.mutex);
    Snapshot copy = c.snap;
    const auto age = std::chrono::steady_clock::now() - c.lastTick;
    copy.valid = c.lastTick.time_since_epoch().count() != 0 && age < std::chrono::milliseconds(750);
    return copy;
}

// ===========================================================================
// Drawing helpers
// ===========================================================================

inline constexpr std::uint32_t kCyan = 0x4DE3FF;
inline constexpr std::uint32_t kCyanDim = 0x2A8FA6;
inline constexpr std::uint32_t kPanelColor = 0x04222B;
inline constexpr std::uint32_t kRed = 0xFF4D4D;
inline constexpr std::uint32_t kAmber = 0xFFC04D;
inline constexpr std::uint32_t kGreen = 0x5CFF9E;
inline constexpr std::uint32_t kWhite = 0xE8FBFF;

inline std::uint32_t argb(float alpha, std::uint32_t rgb) {
    alpha = std::clamp(alpha, 0.f, 1.f);
    return (static_cast<std::uint32_t>(alpha * 255.f) << 24) | (rgb & 0x00FFFFFFu);
}

inline float nowSeconds() {
    using namespace std::chrono;
    return duration<float>(steady_clock::now().time_since_epoch()).count();
}

inline float textWidth(const std::string& text, float size) {
    float width = 0.f;
    for (char ch : text) {
        if (ch == 'i' || ch == 'l' || ch == '1' || ch == ':' || ch == '.' || ch == ' ' || ch == '|') width += size * 0.3f;
        else if (ch == 'm' || ch == 'w' || ch == 'M' || ch == 'W') width += size * 0.8f;
        else width += size * 0.58f;
    }
    return width;
}

inline std::string formatFloat(const char* format, float value) {
    char buffer[48];
    std::snprintf(buffer, sizeof(buffer), format, value);
    return buffer;
}

class Canvas {
public:
    void rect(float x, float y, float w, float h, std::uint32_t color) {
        pl::modmenu::DrawCommand cmd = {};
        cmd.type = pl::modmenu::DrawCommandType::RectFilled;
        cmd.x = x; cmd.y = y; cmd.w = w; cmd.h = h;
        cmd.color = color;
        mCommands.push_back(cmd);
    }

    // The launcher expects (x, y) as the start and (w, h) as the delta.
    void line(float x1, float y1, float x2, float y2, float thickness, std::uint32_t color) {
        pl::modmenu::DrawCommand cmd = {};
        cmd.type = pl::modmenu::DrawCommandType::Line;
        cmd.x = x1; cmd.y = y1; cmd.w = x2 - x1; cmd.h = y2 - y1;
        cmd.size = thickness;
        cmd.color = color;
        mCommands.push_back(cmd);
    }

    void circle(float cx, float cy, float radius, std::uint32_t color) {
        pl::modmenu::DrawCommand cmd = {};
        cmd.type = pl::modmenu::DrawCommandType::CircleFilled;
        cmd.x = cx; cmd.y = cy;
        cmd.size = radius;
        cmd.color = color;
        mCommands.push_back(cmd);
    }

    void triangle(float x1, float y1, float x2, float y2, float x3, float y3, std::uint32_t color) {
        pl::modmenu::DrawCommand cmd = {};
        cmd.type = pl::modmenu::DrawCommandType::TriangleFilled;
        cmd.x = x1; cmd.y = y1; cmd.w = x2; cmd.h = y2; cmd.x3 = x3; cmd.y3 = y3;
        cmd.color = color;
        mCommands.push_back(cmd);
    }

    void text(float x, float y, float size, std::uint32_t color, const std::string& value) {
        if (value.empty()) return;
        mStrings.push_back(value);   // deque: references stay valid until submit()
        pl::modmenu::DrawCommand cmd = {};
        cmd.type = pl::modmenu::DrawCommandType::Text;
        cmd.x = x; cmd.y = y;
        cmd.w = textWidth(value, size);
        cmd.h = size;
        cmd.size = size;
        cmd.color = color;
        cmd.text = mStrings.back().c_str();
        mCommands.push_back(cmd);
    }

    void ring(float cx, float cy, float radius, float thickness, std::uint32_t color, int segments = 48) {
        constexpr float kTwoPi = 6.28318530718f;
        float px = cx + radius, py = cy;
        for (int i = 1; i <= segments; ++i) {
            const float a = kTwoPi * static_cast<float>(i) / static_cast<float>(segments);
            const float nx = cx + std::cos(a) * radius;
            const float ny = cy + std::sin(a) * radius;
            line(px, py, nx, ny, thickness, color);
            px = nx; py = ny;
        }
    }

    // Four corner brackets, like the corners of an Iron Man helmet HUD.
    void brackets(float x, float y, float w, float h, float len, float thickness, std::uint32_t color) {
        line(x, y, x + len, y, thickness, color);
        line(x, y, x, y + len, thickness, color);
        line(x + w, y, x + w - len, y, thickness, color);
        line(x + w, y, x + w, y + len, thickness, color);
        line(x, y + h, x + len, y + h, thickness, color);
        line(x, y + h, x, y + h - len, thickness, color);
        line(x + w, y + h, x + w - len, y + h, thickness, color);
        line(x + w, y + h, x + w, y + h - len, thickness, color);
    }

    void panel(float x, float y, float w, float h, float scale, float opacity) {
        rect(x, y, w, h, argb(opacity * 0.75f, kPanelColor));
        line(x, y, x + w, y, 1.f * scale, argb(opacity * 0.5f, kCyanDim));
        line(x, y + h, x + w, y + h, 1.f * scale, argb(opacity * 0.5f, kCyanDim));
        brackets(x, y, w, h, 10.f * scale, 2.f * scale, argb(std::min(1.f, opacity + 0.3f), kCyan));
    }

    void submit(const std::string& moduleId) { pl::modmenu::submitDrawCommands(moduleId, mCommands); }

private:
    std::vector<pl::modmenu::DrawCommand> mCommands;
    std::deque<std::string> mStrings;
};

struct Threat {
    const char* label;
    std::uint32_t color;
    int level;
};

inline Threat assess(const Snapshot& s) {
    if (s.hostileCount <= 0) return {"CLEAR", kGreen, 0};
    if (s.nearestHostile < 8.f || s.hostileCount >= 4) return {"HIGH", kRed, 3};
    if (s.nearestHostile < 16.f || s.hostileCount >= 2) return {"ELEVATED", kAmber, 2};
    return {"LOW", kCyan, 1};
}

// ===========================================================================
// Panels (one Mod Menu module each)
// ===========================================================================

inline bool parseBool(std::string_view value, bool& out) {
    if (value == "true") { out = true; return true; }
    if (value == "false") { out = false; return true; }
    return false;
}

inline bool parseFloat(std::string_view value, float& out) {
    if (value.empty()) return false;
    const std::string text(value);
    char* end = nullptr;
    const float parsed = std::strtof(text.c_str(), &end);
    if (end == text.c_str() || !std::isfinite(parsed)) return false;
    out = parsed;
    return true;
}

inline bool parseInt(std::string_view value, int& out) {
    if (value.empty()) return false;
    const std::string text(value);
    char* end = nullptr;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str()) return false;
    out = static_cast<int>(parsed);
    return true;
}

struct ConfigEntry {
    std::string key;
    std::string displayName;
    pl::modmenu::ConfigType type;
    std::string defaultValue;
    std::string minValue;
    std::string maxValue;
};

inline ConfigEntry floatEntry(const char* key, const char* name, float value, float minValue, float maxValue) {
    return {key, name, pl::modmenu::ConfigType::SliderFloat, std::to_string(value), std::to_string(minValue), std::to_string(maxValue)};
}

inline ConfigEntry intEntry(const char* key, const char* name, int value, int minValue, int maxValue) {
    return {key, name, pl::modmenu::ConfigType::SliderInt, std::to_string(value), std::to_string(minValue), std::to_string(maxValue)};
}

inline ConfigEntry toggleEntry(const char* key, const char* name, bool value) {
    return {key, name, pl::modmenu::ConfigType::Toggle, value ? "true" : "false", "", ""};
}

class Panel {
public:
    Panel(std::string moduleId, std::string moduleName, std::string moduleDescription, float x, float y)
        : id(std::move(moduleId)), name(std::move(moduleName)), description(std::move(moduleDescription)), hudX(x), hudY(y) {}
    virtual ~Panel() = default;

    std::string id;
    std::string name;
    std::string description;
    bool enabled = false;
    float hudX;
    float hudY;
    float scale = 1.0f;
    float opacity = 0.7f;

    virtual void draw(const Snapshot& snapshot) = 0;
    virtual void onEnabledChanged() {}

    virtual std::vector<ConfigEntry> entries() const {
        return {floatEntry("scale", "Scale", scale, 0.1f, 5.f),
                floatEntry("opacity", "Opacity", opacity, 0.f, 1.f)};
    }

    virtual bool set(const std::string& key, std::string_view value) {
        if (key == "hudPosX") return parseFloat(value, hudX);
        if (key == "hudPosY") return parseFloat(value, hudY);
        if (key == "scale") return parseFloat(value, scale);
        if (key == "opacity") return parseFloat(value, opacity);
        return false;
    }

    virtual void toJson(nlohmann::json& j) const {
        j["enabled"] = enabled;
        j["hudPosX"] = hudX;
        j["hudPosY"] = hudY;
        j["scale"] = scale;
        j["opacity"] = opacity;
    }

    virtual void fromJson(const nlohmann::json& j) {
        if (j.contains("enabled")) enabled = j["enabled"].get<bool>();
        if (j.contains("hudPosX")) hudX = j["hudPosX"].get<float>();
        if (j.contains("hudPosY")) hudY = j["hudPosY"].get<float>();
        if (j.contains("scale")) scale = j["scale"].get<float>();
        if (j.contains("opacity")) opacity = j["opacity"].get<float>();
    }

    void clear() {
        Canvas canvas;
        canvas.submit(id);
    }
};

// ---- Status ---------------------------------------------------------------

class StatusPanel : public Panel {
public:
    StatusPanel()
        : Panel("jarvis.status", "Jarvis Status",
                "Iron Man style status panel: altitude, speed, heading and threat level.", 20.f, 40.f) {}

    bool showSpeed = true;
    bool showHeading = true;

    std::vector<ConfigEntry> entries() const override {
        auto list = Panel::entries();
        list.push_back(toggleEntry("showSpeed", "Show Speed", showSpeed));
        list.push_back(toggleEntry("showHeading", "Show Heading", showHeading));
        return list;
    }

    bool set(const std::string& key, std::string_view value) override {
        if (Panel::set(key, value)) return true;
        if (key == "showSpeed") return parseBool(value, showSpeed);
        if (key == "showHeading") return parseBool(value, showHeading);
        return false;
    }

    void toJson(nlohmann::json& j) const override {
        Panel::toJson(j);
        j["showSpeed"] = showSpeed;
        j["showHeading"] = showHeading;
    }

    void fromJson(const nlohmann::json& j) override {
        Panel::fromJson(j);
        if (j.contains("showSpeed")) showSpeed = j["showSpeed"].get<bool>();
        if (j.contains("showHeading")) showHeading = j["showHeading"].get<bool>();
    }

    void draw(const Snapshot& s) override {
        const float sc = scale;
        const float size = 20.f * sc;
        const float row = size * 1.35f;
        const float pad = 10.f * sc;
        const float gap = 14.f * sc;

        struct Row { std::string label; std::string value; std::uint32_t color; };
        std::vector<Row> rows;
        const Threat threat = assess(s);

        rows.push_back({"ALTITUDE", s.valid ? formatFloat("%.0f m", s.y) : std::string("-- m"), kCyan});
        if (showSpeed) rows.push_back({"SPEED", s.valid ? formatFloat("%.1f m/s", s.speed) : std::string("-- m/s"), kCyan});
        if (showHeading) {
            float bearing = std::fmod(s.yaw + 180.f, 360.f);
            if (bearing < 0.f) bearing += 360.f;
            static const char* names[8] = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"};
            const char* dir = names[static_cast<int>((bearing + 22.5f) / 45.f) % 8];
            char buffer[24];
            std::snprintf(buffer, sizeof(buffer), "%03d %s", static_cast<int>(bearing), dir);
            rows.push_back({"HEADING", s.valid ? std::string(buffer) : std::string("---"), kCyan});
        }
        rows.push_back({"THREAT", s.valid ? std::string(threat.label) : std::string("---"), s.valid ? threat.color : kCyanDim});

        const std::string title = "MARK II";
        std::string status = s.valid ? "J.A.R.V.I.S. ONLINE" : "J.A.R.V.I.S. STANDBY";
        switch (core().state.load()) {
            case 0: status = "WAITING FOR GAME"; break;
            case 2: status = "UNSUPPORTED GAME VERSION"; break;
            case 3: status = "HOOK FAILED"; break;
            default: break;
        }

        float labelW = 0.f, valueW = 0.f;
        for (const auto& r : rows) {
            labelW = std::max(labelW, textWidth(r.label, size));
            valueW = std::max(valueW, textWidth(r.value, size));
        }
        const float headerW = textWidth(title, size) + gap + textWidth(status, size * 0.75f);
        const float width = std::max(labelW + gap + valueW, headerW) + pad * 2.f;
        const float height = pad * 2.f + row * (static_cast<float>(rows.size()) + 1.f);

        Canvas canvas;
        canvas.panel(hudX, hudY, width, height, sc, opacity);

        float y = hudY + pad;
        canvas.text(hudX + pad, y, size, argb(1.f, kWhite), title);
        const float statusX = hudX + width - pad - textWidth(status, size * 0.75f);
        canvas.text(statusX, y + size * 0.2f, size * 0.75f, argb(1.f, s.valid ? kGreen : kAmber), status);
        y += row;

        for (const auto& r : rows) {
            canvas.text(hudX + pad, y, size, argb(0.85f, kCyanDim), r.label);
            canvas.text(hudX + width - pad - textWidth(r.value, size), y, size, argb(1.f, r.color), r.value);
            y += row;
        }
        canvas.submit(id);
    }
};

// ---- Radar ----------------------------------------------------------------

class RadarPanel : public Panel {
public:
    RadarPanel()
        : Panel("jarvis.radar", "Jarvis Radar",
                "Circular radar that tracks nearby hostile mobs, passive mobs and players.", 650.f, 40.f) {}

    float radarRange = 32.f;
    int hostileMask = 4;
    bool showPlayers = true;
    bool showPassive = true;
    bool sweep = true;
    bool debugCategories = false;

    void pushSettings() {
        auto& s = core().settings;
        s.radarRange = radarRange;
        s.hostileMask = hostileMask;
        s.showPlayers = showPlayers;
        s.showPassive = showPassive;
        s.needRadar = enabled;
    }

    void onEnabledChanged() override { pushSettings(); }

    std::vector<ConfigEntry> entries() const override {
        auto list = Panel::entries();
        list.push_back(floatEntry("radarRange", "Radar Range", radarRange, 8.f, 180.f));
        list.push_back(intEntry("hostileMask", "Hostile Mask", hostileMask, 0, 200));
        list.push_back(toggleEntry("showPlayers", "Show Players", showPlayers));
        list.push_back(toggleEntry("showPassive", "Show Passive", showPassive));
        list.push_back(toggleEntry("sweepAnimation", "Sweep Animation", sweep));
        list.push_back(toggleEntry("debugCategories", "Debug Categories", debugCategories));
        return list;
    }

    bool set(const std::string& key, std::string_view value) override {
        bool handled = Panel::set(key, value);
        if (!handled) {
            if (key == "radarRange") handled = parseFloat(value, radarRange);
            else if (key == "hostileMask") handled = parseInt(value, hostileMask);
            else if (key == "showPlayers") handled = parseBool(value, showPlayers);
            else if (key == "showPassive") handled = parseBool(value, showPassive);
            else if (key == "sweepAnimation") handled = parseBool(value, sweep);
            else if (key == "debugCategories") handled = parseBool(value, debugCategories);
        }
        if (handled) pushSettings();
        return handled;
    }

    void toJson(nlohmann::json& j) const override {
        Panel::toJson(j);
        j["radarRange"] = radarRange;
        j["hostileMask"] = hostileMask;
        j["showPlayers"] = showPlayers;
        j["showPassive"] = showPassive;
        j["sweepAnimation"] = sweep;
        j["debugCategories"] = debugCategories;
    }

    void fromJson(const nlohmann::json& j) override {
        Panel::fromJson(j);
        if (j.contains("radarRange")) radarRange = j["radarRange"].get<float>();
        if (j.contains("hostileMask")) hostileMask = j["hostileMask"].get<int>();
        if (j.contains("showPlayers")) showPlayers = j["showPlayers"].get<bool>();
        if (j.contains("showPassive")) showPassive = j["showPassive"].get<bool>();
        if (j.contains("sweepAnimation")) sweep = j["sweepAnimation"].get<bool>();
        if (j.contains("debugCategories")) debugCategories = j["debugCategories"].get<bool>();
        pushSettings();
    }

    void draw(const Snapshot& s) override {
        const float sc = scale;
        const float size = 18.f * sc;
        const float row = size * 1.3f;
        const float pad = 10.f * sc;
        const float radius = 72.f * sc;
        const float range = std::clamp(radarRange, 8.f, 180.f);
        core().settings.debug = debugCategories;
        const Threat threat = assess(s);

        // Threat / debug list under the radar (nearest first, max 4).
        struct ListItem { std::string text; std::uint32_t color; };
        std::vector<ListItem> list;
        for (const auto& c : s.contacts) {
            if (list.size() >= 4) break;
            if (!debugCategories && !c.hostile) continue;
            char buffer[48];
            if (debugCategories) {
                std::snprintf(buffer, sizeof(buffer), "%s %.0f m [0x%X]",
                              c.player ? "PLAYER" : (c.hostile ? "HOSTILE" : "MOB"), c.dist, c.cats);
            } else {
                const std::string label = c.type.empty() ? std::string("HOSTILE") : prettify(c.type);
                std::snprintf(buffer, sizeof(buffer), "%.14s  %.0f m", label.c_str(), c.dist);
            }
            list.push_back({buffer, c.hostile ? kRed : (c.player ? kWhite : kGreen)});
        }

        const float width = radius * 2.f + pad * 2.f;
        const float height = pad * 2.f + row + radius * 2.f + pad +
                             row * static_cast<float>(std::max<std::size_t>(list.size(), 1));

        Canvas canvas;
        canvas.panel(hudX, hudY, width, height, sc, opacity);

        const std::string threats = "THREATS " + std::to_string(s.hostileCount);
        canvas.text(hudX + pad, hudY + pad, size, argb(1.f, kWhite), "RADAR");
        canvas.text(hudX + width - pad - textWidth(threats, size), hudY + pad, size,
                    argb(1.f, s.valid ? threat.color : kCyanDim), threats);

        const float cx = hudX + width * 0.5f;
        const float cy = hudY + pad + row + pad * 0.5f + radius;

        canvas.circle(cx, cy, radius, argb(opacity * 0.35f, kPanelColor));
        canvas.ring(cx, cy, radius, 1.5f * sc, argb(0.9f, kCyan));
        canvas.ring(cx, cy, radius * 0.66f, 1.f * sc, argb(0.35f, kCyan), 36);
        canvas.ring(cx, cy, radius * 0.33f, 1.f * sc, argb(0.35f, kCyan), 28);
        canvas.line(cx - radius, cy, cx + radius, cy, 1.f * sc, argb(0.25f, kCyan));
        canvas.line(cx, cy - radius, cx, cy + radius, 1.f * sc, argb(0.25f, kCyan));

        if (sweep) {
            const float angle = std::fmod(nowSeconds() * 1.6f, 6.28318530718f);
            for (int i = 0; i < 4; ++i) {
                const float a = angle - static_cast<float>(i) * 0.09f;
                canvas.line(cx, cy, cx + std::sin(a) * radius, cy - std::cos(a) * radius,
                            1.5f * sc, argb(0.55f - static_cast<float>(i) * 0.13f, kCyan));
            }
        }

        // Minecraft yaw 0 faces +Z; right-hand side is -X.
        const float yaw = s.yaw * 0.01745329252f;
        const float sinY = std::sin(yaw);
        const float cosY = std::cos(yaw);

        {   // north marker (north is -Z)
            const float r = sinY;
            const float f = -cosY;
            const float nx = cx + r * (radius - size * 0.6f);
            const float ny = cy - f * (radius - size * 0.6f);
            canvas.text(nx - textWidth("N", size * 0.8f) * 0.5f, ny - size * 0.4f, size * 0.8f, argb(1.f, kAmber), "N");
        }

        const float pulse = 0.65f + 0.35f * std::sin(nowSeconds() * 6.f);
        for (const auto& c : s.contacts) {
            if (c.player && !showPlayers) continue;
            if (!c.player && !c.hostile && !showPassive) continue;
            const float right = c.dx * -cosY + c.dz * -sinY;
            const float ahead = c.dx * -sinY + c.dz * cosY;
            if (right * right + ahead * ahead > range * range) continue;
            const float bx = cx + right / range * radius;
            const float by = cy - ahead / range * radius;
            if (c.hostile) {
                canvas.circle(bx, by, 6.f * sc, argb(0.25f * pulse, kRed));
                canvas.circle(bx, by, 3.6f * sc, argb(1.f, kRed));
            } else if (c.player) {
                canvas.circle(bx, by, 3.4f * sc, argb(1.f, kWhite));
            } else {
                canvas.circle(bx, by, 2.8f * sc, argb(0.9f, kGreen));
            }
        }

        canvas.triangle(cx, cy - 6.f * sc, cx - 4.5f * sc, cy + 4.f * sc, cx + 4.5f * sc, cy + 4.f * sc, argb(1.f, kCyan));

        float y = cy + radius + pad;
        if (list.empty()) {
            canvas.text(hudX + pad, y, size, argb(0.8f, kGreen), s.valid ? "NO HOSTILES" : "NO SIGNAL");
        } else {
            for (const auto& item : list) {
                canvas.text(hudX + pad, y, size, argb(1.f, item.color), item.text);
                y += row;
            }
        }

        if (s.valid && threat.level >= 3 && std::fmod(nowSeconds(), 0.8f) < 0.5f) {
            canvas.brackets(hudX - 3.f * sc, hudY - 3.f * sc, width + 6.f * sc, height + 6.f * sc,
                            14.f * sc, 2.5f * sc, argb(1.f, kRed));
        }
        canvas.submit(id);
    }
};

// ---- Scanner --------------------------------------------------------------


// ===========================================================================
// AR overlay helpers (camera math + block knowledge)
// ===========================================================================

struct Camera {
    float x = 0.f, y = 0.f, z = 0.f;
    float rightX = 0.f, rightY = 0.f, rightZ = 0.f;
    float upX = 0.f, upY = 0.f, upZ = 0.f;
    float fwdX = 0.f, fwdY = 0.f, fwdZ = 0.f;
    float focalX = 1.f, focalY = 1.f;   // pixels per unit of (x / depth)
    float halfW = 0.f, halfH = 0.f;
};

// yaw 0 faces +Z, positive pitch looks down (same convention as the radar).
inline Camera makeCamera(const Snapshot& s, float fovDegrees, float width, float height) {
    constexpr float kDeg = 0.017453292519943295f;
    const float yaw = s.yaw * kDeg;
    const float pitch = s.pitch * kDeg;
    Camera cam;
    cam.x = s.x; cam.y = s.y; cam.z = s.z;
    cam.fwdX = -std::sin(yaw) * std::cos(pitch);
    cam.fwdY = -std::sin(pitch);
    cam.fwdZ = std::cos(yaw) * std::cos(pitch);
    cam.rightX = -std::cos(yaw);
    cam.rightY = 0.f;
    cam.rightZ = -std::sin(yaw);
    cam.upX = cam.rightY * cam.fwdZ - cam.rightZ * cam.fwdY;
    cam.upY = cam.rightZ * cam.fwdX - cam.rightX * cam.fwdZ;
    cam.upZ = cam.rightX * cam.fwdY - cam.rightY * cam.fwdX;
    cam.halfW = width * 0.5f;
    cam.halfH = height * 0.5f;
    const float t = std::tan(std::clamp(fovDegrees, 20.f, 140.f) * kDeg * 0.5f);
    cam.focalY = cam.halfH / t;     // vertical field of view
    cam.focalX = cam.focalY;        // square pixels
    return cam;
}

struct CamPoint { float x, y, z; };   // x right, y up, z depth

inline CamPoint toCamera(const Camera& c, float wx, float wy, float wz) {
    const float dx = wx - c.x, dy = wy - c.y, dz = wz - c.z;
    return {dx * c.rightX + dy * c.rightY + dz * c.rightZ,
            dx * c.upX + dy * c.upY + dz * c.upZ,
            dx * c.fwdX + dy * c.fwdY + dz * c.fwdZ};
}

inline void toScreen(const Camera& c, const CamPoint& p, float& sx, float& sy) {
    sx = c.halfW + p.x / p.z * c.focalX;
    sy = c.halfH - p.y / p.z * c.focalY;
}

// Projects a world segment, clipping it against the near plane. False when it is behind us.
inline bool projectSegment(const Camera& c, float ax, float ay, float az, float bx, float by, float bz,
                           float& x1, float& y1, float& x2, float& y2) {
    constexpr float kNear = 0.05f;
    CamPoint a = toCamera(c, ax, ay, az);
    CamPoint b = toCamera(c, bx, by, bz);
    if (a.z < kNear && b.z < kNear) return false;
    if (a.z < kNear) {
        const float t = (kNear - a.z) / (b.z - a.z);
        a = {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, kNear};
    } else if (b.z < kNear) {
        const float t = (kNear - b.z) / (a.z - b.z);
        b = {b.x + (a.x - b.x) * t, b.y + (a.y - b.y) * t, kNear};
    }
    toScreen(c, a, x1, y1);
    toScreen(c, b, x2, y2);
    return std::isfinite(x1) && std::isfinite(y1) && std::isfinite(x2) && std::isfinite(y2);
}

struct BlockInfo {
    const char* key;      // matched against the upper-case block name
    const char* tool;
    float hardness;       // < 0: unbreakable
    float blastResistance;
};

// Rough vanilla values, first match wins. Unknown blocks simply show no numbers.
inline const BlockInfo* blockInfo(const std::string& upperName) {
    static const BlockInfo table[] = {
        {"BEDROCK", "NONE", -1.f, 3600000.f},   {"OBSIDIAN", "PICKAXE", 50.f, 1200.f},
        {"DEEPSLATE", "PICKAXE", 3.f, 6.f},      {"NETHERRACK", "PICKAXE", 0.4f, 0.4f},
        {"ORE", "PICKAXE", 3.f, 3.f},            {"COBBLE", "PICKAXE", 2.f, 6.f},
        {"BRICK", "PICKAXE", 2.f, 6.f},          {"FURNACE", "PICKAXE", 3.5f, 3.5f},
        {"CONCRETE", "PICKAXE", 1.8f, 1.8f},     {"TERRACOTTA", "PICKAXE", 1.25f, 4.2f},
        {"STONE", "PICKAXE", 1.5f, 6.f},         {"ANDESITE", "PICKAXE", 1.5f, 6.f},
        {"DIORITE", "PICKAXE", 1.5f, 6.f},       {"GRANITE", "PICKAXE", 1.5f, 6.f},
        {"IRON BLOCK", "PICKAXE", 5.f, 6.f},     {"CHEST", "AXE", 2.5f, 2.5f},
        {"BARREL", "AXE", 2.5f, 2.5f},           {"CRAFTING", "AXE", 2.5f, 2.5f},
        {"BOOKSHELF", "AXE", 1.5f, 1.5f},        {"LOG", "AXE", 2.f, 2.f},
        {"WOOD", "AXE", 2.f, 2.f},               {"PLANKS", "AXE", 2.f, 3.f},
        {"DOOR", "AXE", 3.f, 3.f},               {"FENCE", "AXE", 2.f, 3.f},
        {"GRASS", "SHOVEL", 0.6f, 0.6f},         {"DIRT", "SHOVEL", 0.5f, 0.5f},
        {"SAND", "SHOVEL", 0.5f, 0.5f},          {"GRAVEL", "SHOVEL", 0.6f, 0.6f},
        {"CLAY", "SHOVEL", 0.6f, 0.6f},          {"SNOW", "SHOVEL", 0.1f, 0.1f},
        {"LEAVES", "SHEARS", 0.2f, 0.2f},        {"WOOL", "SHEARS", 0.8f, 0.8f},
        {"GLASS", "ANY", 0.3f, 0.3f},
    };
    for (const auto& entry : table) {
        if (upperName.find(entry.key) != std::string::npos) return &entry;
    }
    return nullptr;
}

class ScannerPanel : public Panel {
public:
    ScannerPanel()
        : Panel("jarvis.scanner", "Jarvis Scanner",
                "Analyses the block or entity under your crosshair and the item in your hand.", 20.f, 330.f) {}

    bool showTarget = true;
    bool showHeldItem = true;
    bool arOverlay = true;      // box around the target with callouts (world-anchored)
    float fov = 70.f;           // vertical field of view used by the AR projection
    float entityHeight = 1.8f;  // assumed mob height for the AR box

    void pushSettings() {
        core().settings.entityHeight = entityHeight;
        core().settings.arFov = fov;
    }

    void onEnabledChanged() override {
        core().settings.needScanner = enabled;
        pushSettings();
    }

    std::vector<ConfigEntry> entries() const override {
        auto list = Panel::entries();
        list.push_back(toggleEntry("showTarget", "Show Target", showTarget));
        list.push_back(toggleEntry("showHeldItem", "Show Held Item", showHeldItem));
        list.push_back(toggleEntry("arOverlay", "AR Overlay", arOverlay));
        list.push_back(floatEntry("fov", "AR Field Of View", fov, 30.f, 110.f));
        list.push_back(floatEntry("entityHeight", "AR Mob Height", entityHeight, 0.5f, 4.f));
        return list;
    }

    bool set(const std::string& key, std::string_view value) override {
        if (Panel::set(key, value)) return true;
        if (key == "showTarget") return parseBool(value, showTarget);
        if (key == "showHeldItem") return parseBool(value, showHeldItem);
        if (key == "arOverlay") return parseBool(value, arOverlay);
        if (key == "fov") {
            const bool ok = parseFloat(value, fov);
            pushSettings();
            return ok;
        }
        if (key == "entityHeight") {
            const bool ok = parseFloat(value, entityHeight);
            pushSettings();
            return ok;
        }
        return false;
    }

    void toJson(nlohmann::json& j) const override {
        Panel::toJson(j);
        j["showTarget"] = showTarget;
        j["showHeldItem"] = showHeldItem;
        j["arOverlay"] = arOverlay;
        j["fov"] = fov;
        j["entityHeight"] = entityHeight;
    }

    void fromJson(const nlohmann::json& j) override {
        Panel::fromJson(j);
        if (j.contains("showTarget")) showTarget = j["showTarget"].get<bool>();
        if (j.contains("showHeldItem")) showHeldItem = j["showHeldItem"].get<bool>();
        if (j.contains("arOverlay")) arOverlay = j["arOverlay"].get<bool>();
        if (j.contains("fov")) fov = j["fov"].get<float>();
        if (j.contains("entityHeight")) entityHeight = j["entityHeight"].get<float>();
        core().settings.needScanner = enabled;
        pushSettings();
    }

    void draw(const Snapshot& s) override {
        const float sc = scale;
        const float size = 19.f * sc;
        const float row = size * 1.35f;
        const float pad = 10.f * sc;
        const float gap = 14.f * sc;

        struct Line { std::string label; std::string value; std::uint32_t color; };
        std::vector<Line> lines;

        if (showTarget && !arOverlay) {
            const Target& t = s.target;
            if (!s.valid || t.kind == Target::Kind::None) {
                lines.push_back({"TARGET", "NONE", kCyanDim});
            } else if (t.kind == Target::Kind::Block) {
                lines.push_back({"BLOCK", t.name, kCyan});
                lines.push_back({"RANGE", formatFloat("%.1f m", t.dist), kCyan});
            } else {
                const char* cls = t.player ? "PLAYER" : (t.hostile ? "HOSTILE" : "MOB");
                const std::uint32_t color = t.player ? kWhite : (t.hostile ? kRed : kGreen);
                lines.push_back({"ENTITY", cls, color});
                lines.push_back({"RANGE", formatFloat("%.1f m", t.dist), kCyan});
                // Current health is not readable in this game version; show the vanilla maximum.
                float maxHp = 0.f;
                if (!t.player && !t.type.empty() ? mobMaxHealth(t.type, maxHp) : (t.player && mobMaxHealth("player", maxHp))) {
                    lines.push_back({"MAX HP", formatFloat("%.0f", maxHp), kGreen});
                } else {
                    lines.push_back({"HEALTH", "N/A", kCyanDim});
                }
            }
        }
        if (showHeldItem) {
            if (s.valid && s.held.valid) {
                lines.push_back({"HELD", s.held.name, kCyan});
                if (s.held.maxDamage > 0) {
                    const int left = std::max(0, s.held.maxDamage - s.held.damage);
                    const float pct = 100.f * static_cast<float>(left) / static_cast<float>(s.held.maxDamage);
                    char buffer[40];
                    std::snprintf(buffer, sizeof(buffer), "%d/%d (%.0f%%)", left, s.held.maxDamage, pct);
                    lines.push_back({"DURABILITY", buffer, pct < 20.f ? kRed : (pct < 50.f ? kAmber : kGreen)});
                }
            } else {
                lines.push_back({"HELD", "EMPTY", kCyanDim});
            }
        }
        if (lines.empty()) {
            clear();
            return;
        }

        float labelW = 0.f, valueW = 0.f;
        for (const auto& l : lines) {
            labelW = std::max(labelW, textWidth(l.label, size));
            valueW = std::max(valueW, textWidth(l.value, size));
        }
        const std::string title = "SCANNER";
        const float width = std::max(labelW + gap + valueW, textWidth(title, size)) + pad * 2.f;
        const float height = pad * 2.f + row * (static_cast<float>(lines.size()) + 1.f);

        Canvas canvas;
        canvas.panel(hudX, hudY, width, height, sc, opacity);
        float y = hudY + pad;
        canvas.text(hudX + pad, y, size, argb(1.f, kWhite), title);
        y += row;
        for (const auto& l : lines) {
            canvas.text(hudX + pad, y, size, argb(0.85f, kCyanDim), l.label);
            canvas.text(hudX + width - pad - textWidth(l.value, size), y, size, argb(1.f, l.color), l.value);
            y += row;
        }
        if (arOverlay && showTarget) drawOverlay(canvas, s);
        canvas.submit(id);
    }

private:
    void drawOverlay(Canvas& canvas, const Snapshot& s) const {
        const Target& t = s.target;
        if (!s.valid || t.kind == Target::Kind::None || !t.hasBox) return;

        const auto surface = pl::modmenu::getHudSurfaceSize();
        if (!(surface.width > 100.f) || !(surface.height > 100.f)) return;
        const Camera cam = makeCamera(s, fov, surface.width, surface.height);

        const float sc = scale;
        const std::uint32_t lineColor = argb(0.95f, t.kind == Target::Kind::Entity && t.hostile ? kRed : kWhite);

        // Wireframe box.
        const float xs[2] = {t.minX, t.maxX};
        const float ys[2] = {t.minY, t.maxY};
        const float zs[2] = {t.minZ, t.maxZ};
        float left = 1e9f, right = -1e9f, top = 1e9f, bottom = -1e9f;
        bool visible = false;
        auto edge = [&](int ax, int ay, int az, int bx, int by, int bz) {
            float x1, y1, x2, y2;
            if (!projectSegment(cam, xs[ax], ys[ay], zs[az], xs[bx], ys[by], zs[bz], x1, y1, x2, y2)) return;
            canvas.line(x1, y1, x2, y2, 1.6f * sc, lineColor);
            left = std::min({left, x1, x2}); right = std::max({right, x1, x2});
            top = std::min({top, y1, y2}); bottom = std::max({bottom, y1, y2});
            visible = true;
        };
        for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) {
                edge(0, a, b, 1, a, b);   // along X
                edge(a, 0, b, a, 1, b);   // along Y
                edge(a, b, 0, a, b, 1);   // along Z
            }
        }
        if (!visible) return;

        // Callouts: node on the box, a line out, and a small text block.
        const float size = 17.f * sc;
        const float row = size * 1.3f;
        struct Row { std::string text; std::uint32_t color; };
        auto clampX = [&](float v) { return std::clamp(v, 8.f, surface.width - 8.f); };
        auto clampY = [&](float v) { return std::clamp(v, 8.f, surface.height - 8.f); };

        auto callout = [&](float nodeX, float nodeY, float dx, float dy, int align, const std::vector<Row>& rows) {
            if (rows.empty()) return;
            nodeX = clampX(nodeX); nodeY = clampY(nodeY);
            const float endX = clampX(nodeX + dx);
            const float endY = clampY(nodeY + dy);
            canvas.line(nodeX, nodeY, endX, endY, 1.4f * sc, argb(0.9f, kCyan));
            canvas.ring(nodeX, nodeY, 5.f * sc, 1.4f * sc, argb(0.95f, kCyan), 20);
            canvas.ring(endX, endY, 3.f * sc, 1.2f * sc, argb(0.95f, kCyan), 14);

            float width = 0.f;
            for (const auto& r : rows) width = std::max(width, textWidth(r.text, size));
            const float height = row * static_cast<float>(rows.size());
            float x = endX + 6.f * sc;                           // align 1: text to the right of the node
            if (align < 0) x = endX - 6.f * sc - width;          // align -1: text to the left
            else if (align == 0) x = endX - width * 0.5f;        // align 0: centred
            float y = dy < 0.f ? endY - height : endY + 2.f * sc;   // above or below the end node
            x = std::clamp(x, 4.f, surface.width - width - 4.f);
            y = std::clamp(y, 4.f, surface.height - height - 4.f);
            canvas.rect(x - 4.f * sc, y - 2.f * sc, width + 8.f * sc, height + 4.f * sc, argb(0.45f * opacity, kPanelColor));
            for (const auto& r : rows) {
                canvas.text(x, y, size, argb(1.f, r.color), r.text);
                y += row;
            }
        };

        std::vector<Row> topLeft, topRight, bottomRight;
        if (t.kind == Target::Kind::Block) {
            topLeft.push_back({t.name, kWhite});
            topLeft.push_back({formatFloat("RANGE %.1f m", t.dist), kCyan});
            if (const BlockInfo* info = blockInfo(t.name)) {
                if (info->hardness < 0.f) topRight.push_back({"UNBREAKABLE", kRed});
                else topRight.push_back({formatFloat("HARDNESS %.1f", info->hardness), kCyan});
                topRight.push_back({formatFloat("BLAST RES %.1f", info->blastResistance), kCyanDim});
                bottomRight.push_back({std::string("TOOL: ") + info->tool, kAmber});
            }
        } else {
            const char* cls = t.player ? "PLAYER" : (t.hostile ? "HOSTILE" : "MOB");
            const std::uint32_t color = t.player ? kWhite : (t.hostile ? kRed : kGreen);
            topLeft.push_back({t.type.empty() ? std::string(cls) : prettify(t.type), color});
            if (!t.type.empty() && t.hostile) topLeft.push_back({"HOSTILE", kRed});
            if (core().settings.debug.load()) topLeft.push_back({typeFinder().debug.empty() ? std::string("ID: searching") : typeFinder().debug, kCyanDim});
            topLeft.push_back({formatFloat("RANGE %.1f m", t.dist), kCyan});
            float maxHp = 0.f;
            if (t.player ? mobMaxHealth("player", maxHp) : (!t.type.empty() && mobMaxHealth(t.type, maxHp))) {
                topRight.push_back({formatFloat("MAX HP %.0f", maxHp), kGreen});
                topRight.push_back({formatFloat("%.0f HEARTS", maxHp * 0.5f), kCyanDim});
            } else {
                topRight.push_back({"HEALTH N/A", kCyanDim});
            }
            if (t.hostile) bottomRight.push_back({"THREAT: ENGAGE?", kRed});
        }

        callout(left, top, -70.f * sc, -55.f * sc, -1, topLeft);
        callout(right, top, 70.f * sc, -55.f * sc, 1, topRight);
        callout(right, bottom, 55.f * sc, 50.f * sc, 1, bottomRight);
    }
};


// ===========================================================================
// Spectral Glow: every mob around you glows through walls (like a Spectral Arrow)
// ===========================================================================

class GlowPanel : public Panel {
public:
    GlowPanel()
        : Panel("jarvis.glow", "Jarvis Spectral Glow",
                "Outlines mobs with a glowing box that stays visible through walls: red = hostile, green = passive.",
                0.f, 0.f) {}

    float range = 40.f;
    bool hostileOnly = false;
    bool showPlayers = true;
    bool labels = true;
    float strength = 1.f;
    int maxMobs = 20;

    void pushSettings() {
        core().settings.needGlow = enabled;
        core().settings.glowRange = range;
    }

    void onEnabledChanged() override { pushSettings(); }

    std::vector<ConfigEntry> entries() const override {
        auto list = Panel::entries();
        list.push_back(floatEntry("range", "Glow Range (blocks)", range, 10.f, 120.f));
        list.push_back(toggleEntry("hostileOnly", "Hostile Mobs Only", hostileOnly));
        list.push_back(toggleEntry("showPlayers", "Glow Players", showPlayers));
        list.push_back(toggleEntry("labels", "Show Name + Health Labels", labels));
        list.push_back(floatEntry("strength", "Glow Strength", strength, 0.3f, 2.f));
        list.push_back(intEntry("maxMobs", "Max Glowing Mobs", maxMobs, 4, 40));
        return list;
    }

    bool set(const std::string& key, std::string_view value) override {
        if (Panel::set(key, value)) return true;
        if (key == "range") {
            const bool ok = parseFloat(value, range);
            pushSettings();
            return ok;
        }
        if (key == "hostileOnly") return parseBool(value, hostileOnly);
        if (key == "showPlayers") return parseBool(value, showPlayers);
        if (key == "labels") return parseBool(value, labels);
        if (key == "strength") return parseFloat(value, strength);
        if (key == "maxMobs") return parseInt(value, maxMobs);
        return false;
    }

    void toJson(nlohmann::json& j) const override {
        Panel::toJson(j);
        j["range"] = range;
        j["hostileOnly"] = hostileOnly;
        j["showPlayers"] = showPlayers;
        j["labels"] = labels;
        j["strength"] = strength;
        j["maxMobs"] = maxMobs;
    }

    void fromJson(const nlohmann::json& j) override {
        Panel::fromJson(j);
        if (j.contains("range")) range = j["range"].get<float>();
        if (j.contains("hostileOnly")) hostileOnly = j["hostileOnly"].get<bool>();
        if (j.contains("showPlayers")) showPlayers = j["showPlayers"].get<bool>();
        if (j.contains("labels")) labels = j["labels"].get<bool>();
        if (j.contains("strength")) strength = j["strength"].get<float>();
        if (j.contains("maxMobs")) maxMobs = j["maxMobs"].get<int>();
        pushSettings();
    }

    void draw(const Snapshot& s) override {
        const auto surface = pl::modmenu::getHudSurfaceSize();
        if (!s.valid || !(surface.width > 100.f) || !(surface.height > 100.f)) {
            clear();
            return;
        }
        const Camera cam = makeCamera(s, core().settings.arFov.load(), surface.width, surface.height);
        const float sc = scale;
        const float glow = std::clamp(strength, 0.3f, 2.f);
        const float pulse = 0.85f + 0.15f * std::sin(nowSeconds() * 4.f);
        const float labelSize = 15.f * sc;

        Canvas canvas;
        int drawn = 0;
        for (const auto& c : s.contacts) {                 // nearest first
            if (drawn >= std::clamp(maxMobs, 4, 40)) break;
            if (c.dist > range) continue;
            if (hostileOnly && !c.hostile) continue;
            if (c.player && !showPlayers) continue;

            const float xs[2] = {c.wx - c.width * 0.5f, c.wx + c.width * 0.5f};
            const float ys[2] = {c.wy, c.wy + c.height};
            const float zs[2] = {c.wz - c.width * 0.5f, c.wz + c.width * 0.5f};

            struct Seg { float x1, y1, x2, y2; };
            Seg segs[12];
            int count = 0;
            float left = 1e9f, right = -1e9f, top = 1e9f, bottom = -1e9f;
            auto edge = [&](int ax, int ay, int az, int bx, int by, int bz) {
                Seg seg;
                if (!projectSegment(cam, xs[ax], ys[ay], zs[az], xs[bx], ys[by], zs[bz], seg.x1, seg.y1, seg.x2, seg.y2)) return;
                segs[count++] = seg;
                left = std::min({left, seg.x1, seg.x2}); right = std::max({right, seg.x1, seg.x2});
                top = std::min({top, seg.y1, seg.y2}); bottom = std::max({bottom, seg.y1, seg.y2});
            };
            for (int a = 0; a < 2; ++a) {
                for (int b = 0; b < 2; ++b) {
                    edge(0, a, b, 1, a, b);
                    edge(a, 0, b, a, 1, b);
                    edge(a, b, 0, a, b, 1);
                }
            }
            if (count == 0) continue;
            // Entirely off screen: skip it.
            if (right < 0.f || left > surface.width || bottom < 0.f || top > surface.height) continue;
            ++drawn;

            const std::uint32_t color = c.hostile ? kRed : (c.player ? kWhite : kGreen);
            const float fade = std::clamp(1.2f - c.dist / std::max(range, 1.f) * 0.5f, 0.45f, 1.f);

            // Soft body fill (skipped when the mob is right in your face).
            if (c.dist > 2.5f && right - left < surface.width * 0.8f && bottom - top < surface.height * 0.9f) {
                canvas.rect(left, top, right - left, bottom - top, argb(0.07f * glow * pulse * fade, color));
            }
            for (int i = 0; i < count; ++i) {                // outer halo, inner halo, bright core
                canvas.line(segs[i].x1, segs[i].y1, segs[i].x2, segs[i].y2, 6.f * sc, argb(0.10f * glow * pulse * fade, color));
            }
            for (int i = 0; i < count; ++i) {
                canvas.line(segs[i].x1, segs[i].y1, segs[i].x2, segs[i].y2, 3.4f * sc, argb(0.24f * glow * pulse * fade, color));
            }
            for (int i = 0; i < count; ++i) {
                canvas.line(segs[i].x1, segs[i].y1, segs[i].x2, segs[i].y2, 1.4f * sc, argb(std::min(1.f, 0.95f * fade), color));
            }

            if (labels && c.dist <= 45.f) {
                std::string text = c.player ? std::string("PLAYER")
                                            : (c.type.empty() ? std::string(c.hostile ? "HOSTILE" : "MOB") : prettify(c.type));
                text += formatFloat("  %.0fm", c.dist);
                float maxHp = 0.f;
                if (c.player ? mobMaxHealth("player", maxHp) : (!c.type.empty() && mobMaxHealth(c.type, maxHp))) {
                    text += formatFloat("  HP %.0f", maxHp);
                }
                const float w = textWidth(text, labelSize);
                const float x = std::clamp((left + right) * 0.5f - w * 0.5f, 4.f, surface.width - w - 4.f);
                const float y = std::clamp(top - labelSize * 1.5f, 4.f, surface.height - labelSize - 4.f);
                canvas.rect(x - 3.f * sc, y - 1.f * sc, w + 6.f * sc, labelSize * 1.25f, argb(0.45f * opacity, kPanelColor));
                canvas.text(x, y, labelSize, argb(1.f, color), text);
            }
        }
        canvas.submit(id);
    }
};


// ===========================================================================
// J.A.R.V.I.S. AI: asks Claude about the situation around the player
// ===========================================================================
//
// Providers (all plain HTTPS APIs): Groq (free tier), xAI Grok (paid) and Anthropic Claude (paid).
// The HTTPS request goes through java.net.HttpURLConnection (JNI), so the phone's own
// certificate store is used and no TLS library has to be bundled. The only data that
// leaves the device is the text built in buildState() plus the API key header.

struct HttpResult {
    bool ok = false;
    int status = 0;
    std::string body;
    std::string error;
};

template <class VM>
inline jint attachCurrentThread(VM* vm, JNIEnv** env) {
    // Android declares AttachCurrentThread(JNIEnv**, void*), desktop JDKs use void**.
    if constexpr (std::is_invocable_v<decltype(&VM::AttachCurrentThread), VM*, JNIEnv**, void*>) {
        return vm->AttachCurrentThread(env, nullptr);
    } else {
        return vm->AttachCurrentThread(reinterpret_cast<void**>(env), nullptr);
    }
}

inline JavaVM* findJavaVm() {
    using GetVms = jint (*)(JavaVM**, jsize, jsize*);
    void* symbol = dlsym(RTLD_DEFAULT, "JNI_GetCreatedJavaVMs");
    if (!symbol) {
        for (const char* name : {"libnativehelper.so", "libart.so"}) {
            if (void* handle = dlopen(name, RTLD_NOW | RTLD_NOLOAD)) {
                symbol = dlsym(handle, "JNI_GetCreatedJavaVMs");
                if (symbol) break;
            }
        }
    }
    if (!symbol) return nullptr;
    JavaVM* vm = nullptr;
    jsize count = 0;
    if (reinterpret_cast<GetVms>(symbol)(&vm, 1, &count) != JNI_OK || count < 1) return nullptr;
    return vm;
}

// Returns a readable description of the pending Java exception and clears it.
inline std::string takeJavaException(JNIEnv* env) {
    if (!env->ExceptionCheck()) return {};
    jthrowable ex = env->ExceptionOccurred();
    env->ExceptionClear();
    std::string text = "Java exception";
    if (ex) {
        jclass cls = env->FindClass("java/lang/Throwable");
        jmethodID toString = cls ? env->GetMethodID(cls, "toString", "()Ljava/lang/String;") : nullptr;
        if (toString) {
            auto* js = static_cast<jstring>(env->CallObjectMethod(ex, toString));
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
            } else if (js) {
                if (const char* chars = env->GetStringUTFChars(js, nullptr)) {
                    text = chars;
                    env->ReleaseStringUTFChars(js, chars);
                }
            }
        }
        if (env->ExceptionCheck()) env->ExceptionClear();
    }
    return text;
}

inline HttpResult httpPostJson(const std::string& url,
                               const std::vector<std::pair<std::string, std::string>>& headers,
                               const std::string& body, int connectTimeoutMs, int readTimeoutMs) {
    HttpResult result;
    JavaVM* vm = findJavaVm();
    if (!vm) {
        result.error = "Java runtime not reachable";
        return result;
    }

    JNIEnv* env = nullptr;
    bool attached = false;
    const jint envStatus = vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
    if (envStatus == JNI_EDETACHED) {
        if (attachCurrentThread(vm, &env) != JNI_OK || !env) {
            result.error = "Could not attach to the Java runtime";
            return result;
        }
        attached = true;
    } else if (envStatus != JNI_OK || !env) {
        result.error = "Java runtime not available";
        return result;
    }

    struct Cleanup {
        JavaVM* vm; JNIEnv* env; bool attached;
        ~Cleanup() {
            env->PopLocalFrame(nullptr);
            if (attached) vm->DetachCurrentThread();
        }
    };
    env->PushLocalFrame(128);
    Cleanup cleanup{vm, env, attached};

    jobject connection = nullptr;
    auto failed = [&](const char* step) {
        if (!env->ExceptionCheck()) return false;
        result.error = std::string(step) + ": " + takeJavaException(env);
        if (connection) {
            if (jclass huc = env->FindClass("java/net/HttpURLConnection")) {
                if (jmethodID disconnect = env->GetMethodID(huc, "disconnect", "()V")) {
                    env->CallVoidMethod(connection, disconnect);
                    if (env->ExceptionCheck()) env->ExceptionClear();
                }
            }
        }
        return true;
    };

    jclass urlClass = env->FindClass("java/net/URL");
    jclass httpClass = env->FindClass("java/net/HttpURLConnection");
    jclass outClass = env->FindClass("java/io/OutputStream");
    jclass inClass = env->FindClass("java/io/InputStream");
    if (failed("classes") || !urlClass || !httpClass || !outClass || !inClass) {
        if (result.error.empty()) result.error = "Java classes missing";
        return result;
    }

    jmethodID urlInit = env->GetMethodID(urlClass, "<init>", "(Ljava/lang/String;)V");
    jmethodID openConnection = env->GetMethodID(urlClass, "openConnection", "()Ljava/net/URLConnection;");
    jmethodID setMethod = env->GetMethodID(httpClass, "setRequestMethod", "(Ljava/lang/String;)V");
    jmethodID setProperty = env->GetMethodID(httpClass, "setRequestProperty", "(Ljava/lang/String;Ljava/lang/String;)V");
    jmethodID setConnectTimeout = env->GetMethodID(httpClass, "setConnectTimeout", "(I)V");
    jmethodID setReadTimeout = env->GetMethodID(httpClass, "setReadTimeout", "(I)V");
    jmethodID setDoOutput = env->GetMethodID(httpClass, "setDoOutput", "(Z)V");
    jmethodID getOutputStream = env->GetMethodID(httpClass, "getOutputStream", "()Ljava/io/OutputStream;");
    jmethodID getResponseCode = env->GetMethodID(httpClass, "getResponseCode", "()I");
    jmethodID getInputStream = env->GetMethodID(httpClass, "getInputStream", "()Ljava/io/InputStream;");
    jmethodID getErrorStream = env->GetMethodID(httpClass, "getErrorStream", "()Ljava/io/InputStream;");
    jmethodID disconnect = env->GetMethodID(httpClass, "disconnect", "()V");
    jmethodID writeBytes = env->GetMethodID(outClass, "write", "([B)V");
    jmethodID closeOut = env->GetMethodID(outClass, "close", "()V");
    jmethodID readBytes = env->GetMethodID(inClass, "read", "([B)I");
    jmethodID closeIn = env->GetMethodID(inClass, "close", "()V");
    if (failed("methods") || !urlInit || !openConnection || !setMethod || !setProperty || !setConnectTimeout ||
        !setReadTimeout || !setDoOutput || !getOutputStream || !getResponseCode || !getInputStream ||
        !getErrorStream || !disconnect || !writeBytes || !closeOut || !readBytes || !closeIn) {
        if (result.error.empty()) result.error = "Java methods missing";
        return result;
    }

    jstring jurl = env->NewStringUTF(url.c_str());
    jobject urlObject = env->NewObject(urlClass, urlInit, jurl);
    if (failed("url") || !urlObject) {
        if (result.error.empty()) result.error = "Bad URL";
        return result;
    }
    connection = env->CallObjectMethod(urlObject, openConnection);
    if (failed("connect") || !connection) {
        if (result.error.empty()) result.error = "Could not open connection";
        return result;
    }

    env->CallVoidMethod(connection, setMethod, env->NewStringUTF("POST"));
    if (failed("method")) return result;
    env->CallVoidMethod(connection, setConnectTimeout, static_cast<jint>(connectTimeoutMs));
    env->CallVoidMethod(connection, setReadTimeout, static_cast<jint>(readTimeoutMs));
    env->CallVoidMethod(connection, setDoOutput, static_cast<jboolean>(JNI_TRUE));
    if (failed("options")) return result;
    for (const auto& header : headers) {
        env->CallVoidMethod(connection, setProperty, env->NewStringUTF(header.first.c_str()),
                            env->NewStringUTF(header.second.c_str()));
        if (failed("header")) return result;
    }

    jobject out = env->CallObjectMethod(connection, getOutputStream);
    if (failed("request") || !out) {
        if (result.error.empty()) result.error = "No output stream";
        return result;
    }
    jbyteArray payload = env->NewByteArray(static_cast<jsize>(body.size()));
    env->SetByteArrayRegion(payload, 0, static_cast<jsize>(body.size()), reinterpret_cast<const jbyte*>(body.data()));
    env->CallVoidMethod(out, writeBytes, payload);
    if (failed("write")) return result;
    env->CallVoidMethod(out, closeOut);
    if (failed("write")) return result;

    result.status = env->CallIntMethod(connection, getResponseCode);
    if (failed("response")) return result;

    jobject in = env->CallObjectMethod(connection, result.status >= 400 ? getErrorStream : getInputStream);
    if (failed("response body")) return result;
    if (in) {
        jbyteArray chunk = env->NewByteArray(8192);
        std::vector<jbyte> buffer(8192);
        constexpr std::size_t kMaxBody = 256 * 1024;
        while (result.body.size() < kMaxBody) {
            const jint count = env->CallIntMethod(in, readBytes, chunk);
            if (failed("read")) return result;
            if (count < 0) break;
            if (count == 0) continue;
            env->GetByteArrayRegion(chunk, 0, count, buffer.data());
            result.body.append(reinterpret_cast<const char*>(buffer.data()), static_cast<std::size_t>(count));
        }
        env->CallVoidMethod(in, closeIn);
        if (env->ExceptionCheck()) env->ExceptionClear();
    }
    env->CallVoidMethod(connection, disconnect);
    if (env->ExceptionCheck()) env->ExceptionClear();
    result.ok = result.status >= 200 && result.status < 300;
    if (!result.ok && result.error.empty()) result.error = "HTTP " + std::to_string(result.status);
    return result;
}

// ---- prompt building -------------------------------------------------------

inline const char* compassName(float dx, float dz) {
    constexpr float kRad = 57.29577951308232f;
    float degrees = std::atan2(dx, -dz) * kRad;   // north is -Z, east is +X
    if (degrees < 0.f) degrees += 360.f;
    static const char* names[] = {"north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"};
    return names[static_cast<int>((degrees + 22.5f) / 45.f) % 8];
}

inline const char* headingName(float yawDegrees) {
    float bearing = std::fmod(yawDegrees + 180.f, 360.f);   // yaw 0 faces south
    if (bearing < 0.f) bearing += 360.f;
    static const char* names[] = {"north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"};
    return names[static_cast<int>((bearing + 22.5f) / 45.f) % 8];
}

inline const char* relativeName(float dx, float dz, float yawDegrees) {
    constexpr float kDeg = 0.017453292519943295f;
    constexpr float kRad = 57.29577951308232f;
    const float yaw = yawDegrees * kDeg;
    const float front = dx * (-std::sin(yaw)) + dz * std::cos(yaw);
    const float side = dx * (-std::cos(yaw)) + dz * (-std::sin(yaw));
    float degrees = std::atan2(side, front) * kRad;   // 0 ahead, 90 to the right
    if (degrees < 0.f) degrees += 360.f;
    static const char* names[] = {"ahead", "ahead-right", "right", "behind-right", "behind", "behind-left", "left", "ahead-left"};
    return names[static_cast<int>((degrees + 22.5f) / 45.f) % 8];
}

inline std::string buildState(const Snapshot& s) {
    std::string out;
    char line[200];
    if (!s.valid) return "No live game data (the player is in a menu or loading).";

    std::snprintf(line, sizeof(line), "PLAYER: position %.0f %.0f %.0f, facing %s, speed %.1f m/s\n", s.x, s.y, s.z,
                  headingName(s.yaw), s.speed);
    out += line;

    std::snprintf(line, sizeof(line), "HOSTILES: %d", s.hostileCount);
    out += line;
    if (s.hostileCount > 0 && s.nearestHostile >= 0.f) {
        std::snprintf(line, sizeof(line), " (nearest %.0f m)", s.nearestHostile);
        out += line;
    }
    out += "\nNEARBY CONTACTS (nearest first):\n";
    int listed = 0;
    for (const auto& contact : s.contacts) {
        if (listed >= 12) break;
        const std::string kindText = contact.type.empty()
            ? std::string(contact.player ? "PLAYER" : (contact.hostile ? "HOSTILE" : "PASSIVE"))
            : prettify(contact.type) + (contact.hostile ? " (hostile)" : "");
        const char* kind = kindText.c_str();
        const char* vertical = contact.dy > 3.f ? ", above you" : (contact.dy < -3.f ? ", below you" : "");
        std::snprintf(line, sizeof(line), "- %s, %.0f m, %s (%s)%s\n", kind, contact.dist,
                      relativeName(contact.dx, contact.dz, s.yaw), compassName(contact.dx, contact.dz), vertical);
        out += line;
        ++listed;
    }
    if (listed == 0) out += "- none\n";

    out += "CROSSHAIR: ";
    if (s.target.kind == Target::Kind::Block) {
        std::snprintf(line, sizeof(line), "block %s, %.1f m\n", s.target.name.c_str(), s.target.dist);
        out += line;
    } else if (s.target.kind == Target::Kind::Entity) {
        const std::string who = !s.target.type.empty() ? prettify(s.target.type)
                                : (s.target.player ? "player" : (s.target.hostile ? "hostile mob" : "mob"));
        std::snprintf(line, sizeof(line), "%s%s, %.1f m\n", who.c_str(), s.target.hostile ? " (hostile)" : "", s.target.dist);
        out += line;
    } else {
        out += "nothing\n";
    }

    out += "HELD ITEM: ";
    if (s.held.valid) {
        out += s.held.name;
        if (s.held.maxDamage > 0) {
            const int left = std::max(0, s.held.maxDamage - s.held.damage);
            std::snprintf(line, sizeof(line), " (durability %d of %d)", left, s.held.maxDamage);
            out += line;
        }
        out += "\n";
    } else {
        out += "empty hand\n";
    }
    out += "Health values of the player and of mobs are not available.";
    return out;
}

// Keeps the HUD font happy: ASCII only, no markdown decoration.
inline std::string cleanAnswer(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < text.size(); ++i) {
        const unsigned char ch = static_cast<unsigned char>(text[i]);
        if (ch == '\r' || ch == '*' || ch == '#' || ch == '`') continue;
        if (ch == '\n' || (ch >= 32 && ch < 127)) {
            out.push_back(static_cast<char>(ch));
            continue;
        }
        if (ch < 0x80) continue;
        if ((ch & 0xC0) == 0x80) continue;   // stray continuation byte

        // Decode one UTF-8 character and map the common typographic ones to ASCII.
        std::size_t length = (ch & 0xE0) == 0xC0 ? 2 : ((ch & 0xF0) == 0xE0 ? 3 : 4);
        std::uint32_t cp = length == 2 ? (ch & 0x1Fu) : (length == 3 ? (ch & 0x0Fu) : (ch & 0x07u));
        std::size_t used = 1;
        while (used < length && i + used < text.size() &&
               (static_cast<unsigned char>(text[i + used]) & 0xC0) == 0x80) {
            cp = (cp << 6) | (static_cast<unsigned char>(text[i + used]) & 0x3Fu);
            ++used;
        }
        i += used - 1;
        if (cp == 0xA0 || cp == 0x202F || cp == 0x2009 || cp == 0x200A || cp == 0x2007) out.push_back(' ');
        else if (cp == 0x200B || cp == 0x200C || cp == 0x200D || cp == 0xFEFF) continue;
        else if (cp == 0x2013 || cp == 0x2014 || cp == 0x2212) out.push_back('-');
        else if (cp == 0x2018 || cp == 0x2019) out.push_back('\'');
        else if (cp == 0x201C || cp == 0x201D) out.push_back('"');
        else if (cp == 0x2026) out += "...";
        else if (cp == 0x2192) out += "->";
        else if (cp == 0x2022) out.push_back('-');
        else out.push_back('?');
    }
    return out;
}

inline std::string trimCopy(std::string_view value) {
    std::size_t a = 0, b = value.size();
    while (a < b && std::isspace(static_cast<unsigned char>(value[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(value[b - 1]))) --b;
    return std::string(value.substr(a, b - a));
}

// ---- background worker -----------------------------------------------------

// ---- text to speech (android.speech.tts.TextToSpeech through JNI) ----------

struct VoiceService {
    std::mutex mutex;
    std::condition_variable wake;
    std::thread worker;
    bool stop = false;
    bool hasText = false;
    std::string text;
    std::string language;
    std::string error;   // guarded by mutex
};

inline VoiceService& voiceService() {
    static VoiceService instance;
    return instance;
}

inline std::string voiceError() {
    auto& v = voiceService();
    std::lock_guard<std::mutex> lock(v.mutex);
    return v.error;
}

inline void voiceLoop() {
    auto& v = voiceService();
    auto setError = [&](const std::string& message) {
        std::lock_guard<std::mutex> lock(v.mutex);
        v.error = message;
    };

    JavaVM* vm = findJavaVm();
    if (!vm) { setError("VOICE: Java runtime not reachable"); return; }
    JNIEnv* env = nullptr;
    bool attached = false;
    const jint envStatus = vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
    if (envStatus == JNI_EDETACHED) {
        if (attachCurrentThread(vm, &env) != JNI_OK || !env) { setError("VOICE: could not attach to Java"); return; }
        attached = true;
    } else if (envStatus != JNI_OK || !env) {
        setError("VOICE: Java runtime not available");
        return;
    }

    jobject tts = nullptr;                        // global reference
    std::string currentLanguage;
    std::chrono::steady_clock::time_point readyAt{};

    auto speakOnce = [&](const std::string& text, const std::string& language) {
        auto failed = [&](const char* step) {
            if (!env->ExceptionCheck()) return false;
            setError(std::string("VOICE ") + step + ": " + takeJavaException(env));
            return true;
        };

        if (!tts) {
            jclass threadClass = env->FindClass("android/app/ActivityThread");
            if (failed("context") || !threadClass) { setError("VOICE: no Android context"); return; }
            jmethodID currentApplication = env->GetStaticMethodID(threadClass, "currentApplication", "()Landroid/app/Application;");
            if (failed("context") || !currentApplication) { setError("VOICE: no Android context"); return; }
            jobject application = env->CallStaticObjectMethod(threadClass, currentApplication);
            if (failed("context") || !application) { setError("VOICE: no Android context"); return; }
            jclass ttsClass = env->FindClass("android/speech/tts/TextToSpeech");
            if (failed("engine") || !ttsClass) { setError("VOICE: no text-to-speech engine class"); return; }
            jmethodID constructor = env->GetMethodID(ttsClass, "<init>",
                                                     "(Landroid/content/Context;Landroid/speech/tts/TextToSpeech$OnInitListener;)V");
            if (failed("engine") || !constructor) { setError("VOICE: no text-to-speech engine"); return; }
            jobject created = env->NewObject(ttsClass, constructor, application, static_cast<jobject>(nullptr));
            if (failed("engine") || !created) { setError("VOICE: could not start text-to-speech"); return; }
            tts = env->NewGlobalRef(created);
            readyAt = std::chrono::steady_clock::now() + std::chrono::milliseconds(1500);
            currentLanguage.clear();
        }

        const auto now = std::chrono::steady_clock::now();
        if (now < readyAt) std::this_thread::sleep_for(readyAt - now);

        jclass ttsClass = env->GetObjectClass(tts);
        jclass localeClass = env->FindClass("java/util/Locale");
        jmethodID forTag = localeClass ? env->GetStaticMethodID(localeClass, "forLanguageTag", "(Ljava/lang/String;)Ljava/util/Locale;") : nullptr;
        jmethodID setLanguage = env->GetMethodID(ttsClass, "setLanguage", "(Ljava/util/Locale;)I");
        jmethodID speak = env->GetMethodID(ttsClass, "speak", "(Ljava/lang/CharSequence;ILandroid/os/Bundle;Ljava/lang/String;)I");
        if (failed("methods") || !forTag || !setLanguage || !speak) { setError("VOICE: text-to-speech API missing"); return; }

        jint languageResult = 0;
        for (int attempt = 0; attempt < 5; ++attempt) {
            if (currentLanguage != language) {
                jobject locale = env->CallStaticObjectMethod(localeClass, forTag, env->NewStringUTF(language.c_str()));
                if (failed("language")) return;
                languageResult = env->CallIntMethod(tts, setLanguage, locale);
                if (failed("language")) return;
                if (languageResult >= 0) currentLanguage = language;
            }
            if (currentLanguage == language) {
                const jint result = env->CallIntMethod(tts, speak, env->NewStringUTF(text.c_str()), 0,
                                                       static_cast<jobject>(nullptr), env->NewStringUTF("jarvis"));
                if (failed("speak")) return;
                if (result == 0) {
                    setError({});
                    return;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(700));   // engine still starting up
        }
        setError(currentLanguage == language ? "VOICE: text-to-speech engine not ready"
                                             : "VOICE: language not available (" + language + ")");
    };

    for (;;) {
        std::string text, language;
        {
            std::unique_lock<std::mutex> lock(v.mutex);
            v.wake.wait(lock, [&] { return v.stop || v.hasText; });
            if (v.stop) break;
            text = std::move(v.text);
            language = std::move(v.language);
            v.hasText = false;
        }
        env->PushLocalFrame(32);
        speakOnce(text, language);
        env->PopLocalFrame(nullptr);
    }

    if (tts) {
        if (jclass ttsClass = env->GetObjectClass(tts)) {
            if (jmethodID shutdown = env->GetMethodID(ttsClass, "shutdown", "()V")) env->CallVoidMethod(tts, shutdown);
        }
        if (env->ExceptionCheck()) env->ExceptionClear();
        env->DeleteGlobalRef(tts);
    }
    if (attached) vm->DetachCurrentThread();
}

inline void voiceSpeak(const std::string& text, const std::string& language) {
    if (text.empty()) return;
    auto& v = voiceService();
    {
        std::lock_guard<std::mutex> lock(v.mutex);
        v.text = text;
        v.language = language.empty() ? std::string("en-US") : language;
        v.hasText = true;
        v.stop = false;
        if (!v.worker.joinable()) v.worker = std::thread(voiceLoop);
    }
    v.wake.notify_one();
}

inline void voiceShutdown() {
    auto& v = voiceService();
    {
        std::lock_guard<std::mutex> lock(v.mutex);
        v.stop = true;
    }
    v.wake.notify_all();
    if (v.worker.joinable()) v.worker.join();
}

struct AiJob {
    std::string provider;   // groq (free tier), grok (xAI, paid) or claude (Anthropic, paid)
    std::string apiKey;
    std::string model;
    std::string language;
    std::string question;   // empty: threat scan
    bool speak = false;
    std::string voiceLanguage;
    bool proactive = false; // started by Jarvis itself (question holds the event description)
};

struct AiService {
    std::mutex mutex;
    std::condition_variable wake;
    std::thread worker;
    bool stop = false;
    bool hasJob = false;
    AiJob job;

    std::atomic<bool> busy{false};
    std::string answer;                 // guarded by mutex
    std::string error;                  // guarded by mutex
    bool answerAuto = false;            // guarded by mutex: the answer was Jarvis' own initiative
    std::chrono::steady_clock::time_point answerAt{};
    std::chrono::steady_clock::time_point lastRequest{};
};

inline AiService& ai() {
    static AiService instance;
    return instance;
}

struct ProviderInfo {
    std::string url;
    std::string defaultModel;
    bool anthropic = false;
};

inline ProviderInfo providerInfo(std::string name) {
    for (auto& ch : name) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (name == "claude" || name == "anthropic") {
        return {"https://api.anthropic.com/v1/messages", "claude-haiku-5-5", true};
    }
    if (name == "grok" || name == "xai") {
        return {"https://api.x.ai/v1/chat/completions", "grok-4.20-0309-non-reasoning", false};
    }
    return {"https://api.groq.com/openai/v1/chat/completions", "openai/gpt-oss-20b", false};   // default: Groq
}

struct AiRequest {
    std::string url;
    std::vector<std::pair<std::string, std::string>> headers;
    std::string body;
};

inline AiRequest buildAiRequest(const AiJob& job, const Snapshot& snap) {
    const ProviderInfo info = providerInfo(job.provider);
    const std::string model = job.model.empty() ? info.defaultModel : job.model;
    const std::string language = job.language.empty() ? std::string("English") : job.language;
    const std::string system =
        "You are J.A.R.V.I.S., the tactical assistant in the heads-up display of a Minecraft Bedrock player. "
        "Use only the data in the message; never invent mobs, blocks, items or health values. "
        "Answer in at most 55 words and at most 4 short lines, as plain ASCII text without markdown or emoji. "
        "Reply in " + language + ".";
    std::string user;
    if (job.proactive) {
        user = "AUTOMATIC ALERT (the player did not ask anything): " + job.question +
               "\nGive a proactive heads-up in at most 35 words: what it is, how dangerous it is, and one concrete tip.\n\n";
    } else if (job.question.empty()) {
        user = "Give a short threat assessment of the surroundings and one recommended action.\n\n";
    } else {
        user = "Player question: " + job.question + "\n\n";
    }
    user += buildState(snap);

    AiRequest request;
    request.url = info.url;
    request.headers = {{"content-type", "application/json"}, {"user-agent", "JarvisMod/1.0"}};
    nlohmann::json body;
    body["model"] = model;
    if (info.anthropic) {
        request.headers.emplace_back("x-api-key", job.apiKey);
        request.headers.emplace_back("anthropic-version", "2023-06-01");
        body["max_tokens"] = 300;
        body["system"] = system;
        body["messages"] = nlohmann::json::array({{{"role", "user"}, {"content", user}}});
    } else {
        request.headers.emplace_back("authorization", "Bearer " + job.apiKey);
        body["max_completion_tokens"] = 400;
        body["temperature"] = 0.4;
        if (model.find("gpt-oss") != std::string::npos) body["reasoning_effort"] = "low";
        body["messages"] = nlohmann::json::array({{{"role", "system"}, {"content", system}},
                                                  {{"role", "user"}, {"content", user}}});
    }
    request.body = body.dump();
    return request;
}

// Pulls the answer (or a readable error) out of the HTTP result. Returns true on success.
inline bool parseAiResponse(const std::string& provider, const HttpResult& response, std::string& text, std::string& error) {
    if (response.status == 0) {
        error = "NETWORK: " + response.error;
        return false;
    }
    const nlohmann::json parsed = nlohmann::json::parse(response.body, nullptr, false);
    if (!response.ok) {
        std::string message = response.error;
        if (parsed.is_object() && parsed.contains("error")) {
            const auto& err = parsed["error"];
            if (err.is_object() && err.contains("message") && err["message"].is_string()) {
                message = err["message"].get<std::string>();
            } else if (err.is_string()) {
                message = err.get<std::string>();
            }
        }
        if (response.status == 401 || response.status == 403) message = "Invalid API key or access denied";
        if (response.status == 429) message = "Rate limit reached, try again in a moment";
        error = "ERROR " + std::to_string(response.status) + ": " + cleanAnswer(message);
        return false;
    }

    std::string out;
    if (providerInfo(provider).anthropic) {
        if (parsed.is_object() && parsed.contains("content") && parsed["content"].is_array()) {
            for (const auto& block : parsed["content"]) {
                if (block.is_object() && block.value("type", "") == "text" && block.contains("text") && block["text"].is_string()) {
                    out += block["text"].get<std::string>();
                }
            }
        }
    } else if (parsed.is_object() && parsed.contains("choices") && parsed["choices"].is_array() && !parsed["choices"].empty()) {
        const auto& first = parsed["choices"][0];
        if (first.is_object() && first.contains("message") && first["message"].is_object() &&
            first["message"].contains("content") && first["message"]["content"].is_string()) {
            out = first["message"]["content"].get<std::string>();
        }
    }
    text = cleanAnswer(trimCopy(out));
    if (text.empty()) {
        error = "The AI sent an empty answer (try another model)";
        return false;
    }
    return true;
}

inline void aiRunJob(const AiJob& job) {
    auto& a = ai();
    const Snapshot snap = snapshot();
    const AiRequest request = buildAiRequest(job, snap);
    const HttpResult response = httpPostJson(request.url, request.headers, request.body, 10000, 30000);

    std::string text, error;
    const bool ok = parseAiResponse(job.provider, response, text, error);
    if (!ok && job.proactive) return;   // an automatic alert that fails stays silent (no red error)
    {
        std::lock_guard<std::mutex> lock(a.mutex);
        a.answer = ok ? text : std::string();
        a.error = ok ? std::string() : error;
        a.answerAuto = job.proactive;
        a.answerAt = std::chrono::steady_clock::now();
    }
    if (ok && job.speak) voiceSpeak(text, job.voiceLanguage);
}

inline void aiWorkerLoop() {
    auto& a = ai();
    for (;;) {
        AiJob job;
        {
            std::unique_lock<std::mutex> lock(a.mutex);
            a.wake.wait(lock, [&] { return a.stop || a.hasJob; });
            if (a.stop) return;
            job = std::move(a.job);
            a.hasJob = false;
        }
        a.busy = true;
        try {
            aiRunJob(job);
        } catch (...) {
            std::lock_guard<std::mutex> lock(a.mutex);
            a.error = "Unexpected error";
            a.answerAt = std::chrono::steady_clock::now();
        }
        a.busy = false;
    }
}

// Returns false when the request was ignored (busy, too soon, no key).
inline bool aiRequest(const AiJob& job, std::string& reason) {
    auto& a = ai();
    if (job.apiKey.empty()) {
        std::lock_guard<std::mutex> lock(a.mutex);
        a.error = "Set your API key in the Jarvis AI settings first";
        a.answer.clear();
        a.answerAt = std::chrono::steady_clock::now();
        reason = "no key";
        return false;
    }
    const auto now = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(a.mutex);
    if (a.busy.load() || a.hasJob) { reason = "busy"; return false; }
    if (a.lastRequest.time_since_epoch().count() != 0 && now - a.lastRequest < std::chrono::seconds(2)) {
        reason = "too soon";
        return false;
    }
    a.lastRequest = now;
    a.job = job;
    a.hasJob = true;
    a.stop = false;
    a.busy = true;   // show "thinking" right away
    if (!a.worker.joinable()) a.worker = std::thread(aiWorkerLoop);
    lock.unlock();
    a.wake.notify_one();
    return true;
}

inline void aiShutdown() {
    auto& a = ai();
    {
        std::lock_guard<std::mutex> lock(a.mutex);
        a.stop = true;
    }
    a.wake.notify_all();
    if (a.worker.joinable()) a.worker.join();
    a.busy = false;
    a.hasJob = false;
}

inline std::vector<std::string> wrapText(const std::string& text, float size, float maxWidth, std::size_t maxLines) {
    std::vector<std::string> lines;
    std::string current, word;
    auto flushWord = [&]() {
        if (word.empty()) return;
        const std::string candidate = current.empty() ? word : current + " " + word;
        if (current.empty() || textWidth(candidate, size) <= maxWidth) {
            current = candidate;
        } else {
            lines.push_back(current);
            current = word;
        }
        word.clear();
    };
    for (char ch : text) {
        if (ch == '\n') {
            flushWord();
            lines.push_back(current);
            current.clear();
        } else if (ch == ' ') {
            flushWord();
        } else {
            word.push_back(ch);
        }
    }
    flushWord();
    if (!current.empty()) lines.push_back(current);
    if (lines.size() > maxLines) {
        lines.resize(maxLines);
        lines.back() += "...";
    }
    return lines;
}

class AiPanel : public Panel {
public:
    AiPanel()
        : Panel("jarvis.ai", "Jarvis AI",
                "Ask an AI about your surroundings. Needs an internet connection and an API key (Groq has a free one).",
                1000.f, 40.f) {}

    std::string provider = "groq";   // groq (free tier), grok (xAI) or claude (Anthropic)
    std::string apiKey;
    std::string model;               // empty: the provider's default model
    std::string language = "English";
    float showSeconds = 45.f;
    float panelWidth = 340.f;
    bool autoReport = true;          // announce the mob under the crosshair and new hostiles nearby
    bool voice = true;               // read reports and AI answers aloud
    std::string voiceLang = "en-US"; // BCP-47 tag: en-US, id-ID ...
    float warnRadius = 24.f;         // hostiles inside this distance trigger a warning
    bool autoInsight = true;         // Jarvis asks the AI by itself when something important happens
    float insightCooldown = 30.f;    // minimum seconds between automatic AI insights
    std::string quickQuestion = "What am I looking at, and what should I do?";   // sent by the Ask Jarvis button

    bool wantsGameData() const { return enabled && (autoReport || autoInsight); }

    void onEnabledChanged() override { core().settings.aiNeeds = wantsGameData(); }

    std::vector<ConfigEntry> entries() const override {
        auto list = Panel::entries();
        list.push_back({"provider", "Provider (groq / grok / claude)", pl::modmenu::ConfigType::Text, provider, "", ""});
        list.push_back({"apiKey", "API Key", pl::modmenu::ConfigType::Text, apiKey, "", ""});
        list.push_back({"scan", "Scan Surroundings (ask Jarvis)", pl::modmenu::ConfigType::Button, "false", "", ""});
        list.push_back({"question", "Ask Jarvis (type a question)", pl::modmenu::ConfigType::Text, "", "", ""});
        list.push_back({"model", "Model (empty = default)", pl::modmenu::ConfigType::Text, model, "", ""});
        list.push_back({"language", "Reply Language", pl::modmenu::ConfigType::Text, language, "", ""});
        list.push_back({"quickQuestion", "Ask Jarvis Button Question", pl::modmenu::ConfigType::Text, quickQuestion, "", ""});
        list.push_back(toggleEntry("autoReport", "Auto Report Mobs", autoReport));
        list.push_back(toggleEntry("autoInsight", "Auto AI Insights (proactive)", autoInsight));
        list.push_back(floatEntry("insightCooldown", "Insight Cooldown (seconds)", insightCooldown, 10.f, 180.f));
        list.push_back(toggleEntry("voice", "Voice (speak aloud)", voice));
        list.push_back({"voiceLang", "Voice Language (en-US / id-ID)", pl::modmenu::ConfigType::Text, voiceLang, "", ""});
        list.push_back(floatEntry("warnRadius", "Warning Radius", warnRadius, 8.f, 80.f));
        list.push_back(floatEntry("showSeconds", "Show Answer (seconds)", showSeconds, 5.f, 300.f));
        list.push_back(floatEntry("panelWidth", "Panel Width", panelWidth, 200.f, 700.f));
        return list;
    }

    bool set(const std::string& key, std::string_view value) override {
        if (Panel::set(key, value)) return true;
        if (key == "provider") { provider = trimCopy(value); if (provider.empty()) provider = "groq"; return true; }
        if (key == "apiKey") { apiKey = trimCopy(value); return true; }
        if (key == "model") { model = trimCopy(value); return true; }
        if (key == "language") { language = trimCopy(value); return true; }
        if (key == "showSeconds") return parseFloat(value, showSeconds);
        if (key == "panelWidth") return parseFloat(value, panelWidth);
        if (key == "autoReport") {
            const bool ok = parseBool(value, autoReport);
            core().settings.aiNeeds = wantsGameData();
            return ok;
        }
        if (key == "autoInsight") {
            const bool ok = parseBool(value, autoInsight);
            core().settings.aiNeeds = wantsGameData();
            return ok;
        }
        if (key == "insightCooldown") return parseFloat(value, insightCooldown);
        if (key == "quickQuestion") {
            quickQuestion = trimCopy(value);
            if (quickQuestion.empty()) quickQuestion = "What am I looking at, and what should I do?";
            return true;
        }
        if (key == "voice") return parseBool(value, voice);
        if (key == "voiceLang") { voiceLang = trimCopy(value); if (voiceLang.empty()) voiceLang = "en-US"; return true; }
        if (key == "warnRadius") return parseFloat(value, warnRadius);
        if (key == "question") {
            pending = trimCopy(value);
            pendingAt = std::chrono::steady_clock::now();
            return false;   // never saved
        }
        if (key == "scan") {
            // Ignore the callbacks the menu may fire while it registers the control.
            if (std::chrono::steady_clock::now() - createdAt > std::chrono::seconds(4)) scan();
            return false;
        }
        return false;
    }

    void toJson(nlohmann::json& j) const override {
        Panel::toJson(j);
        j["provider"] = provider;
        j["apiKey"] = apiKey;
        j["model"] = model;
        j["language"] = language;
        j["showSeconds"] = showSeconds;
        j["panelWidth"] = panelWidth;
        j["autoReport"] = autoReport;
        j["voice"] = voice;
        j["voiceLang"] = voiceLang;
        j["warnRadius"] = warnRadius;
        j["autoInsight"] = autoInsight;
        j["insightCooldown"] = insightCooldown;
        j["quickQuestion"] = quickQuestion;
    }

    void fromJson(const nlohmann::json& j) override {
        Panel::fromJson(j);
        if (j.contains("provider") && j["provider"].is_string()) provider = j["provider"].get<std::string>();
        if (j.contains("apiKey") && j["apiKey"].is_string()) apiKey = j["apiKey"].get<std::string>();
        if (j.contains("model") && j["model"].is_string()) model = j["model"].get<std::string>();
        if (j.contains("language") && j["language"].is_string()) language = j["language"].get<std::string>();
        if (j.contains("showSeconds")) showSeconds = j["showSeconds"].get<float>();
        if (j.contains("panelWidth")) panelWidth = j["panelWidth"].get<float>();
        if (j.contains("autoReport")) autoReport = j["autoReport"].get<bool>();
        if (j.contains("voice")) voice = j["voice"].get<bool>();
        if (j.contains("voiceLang") && j["voiceLang"].is_string()) voiceLang = j["voiceLang"].get<std::string>();
        if (j.contains("warnRadius")) warnRadius = j["warnRadius"].get<float>();
        if (j.contains("autoInsight")) autoInsight = j["autoInsight"].get<bool>();
        if (j.contains("insightCooldown")) insightCooldown = j["insightCooldown"].get<float>();
        if (j.contains("quickQuestion") && j["quickQuestion"].is_string() && !j["quickQuestion"].get<std::string>().empty()) {
            quickQuestion = j["quickQuestion"].get<std::string>();
        }
        core().settings.aiNeeds = wantsGameData();
    }

    void scan() { submit({}); }

    // The on-screen "Ask Jarvis" button: one tap, no typing.
    void ask() { submit(quickQuestion); }

    void submit(const std::string& question) {
        AiJob job;
        job.provider = provider;
        job.apiKey = apiKey;
        job.model = model;
        job.language = language;
        job.question = question;
        job.speak = voice;
        job.voiceLanguage = voiceLang;
        std::string reason;
        aiRequest(job, reason);
    }

    void draw(const Snapshot& s) override {
        const auto now = std::chrono::steady_clock::now();

        // A typed question is sent once the text has stopped changing for a moment.
        if (!pending.empty() && now - pendingAt > std::chrono::milliseconds(1800)) {
            const std::string question = pending;
            pending.clear();
            submit(question);
        }
        if (autoReport || autoInsight) autoReports(s, now);

        auto& a = ai();
        const bool busy = a.busy.load();
        std::string answer, error;
        bool answerAuto = false;
        std::chrono::steady_clock::time_point answerAt;
        {
            std::lock_guard<std::mutex> lock(a.mutex);
            answer = a.answer;
            error = a.error;
            answerAuto = a.answerAuto;
            answerAt = a.answerAt;
        }

        const bool aiFresh = answerAt.time_since_epoch().count() != 0 &&
                             std::chrono::duration<float>(now - answerAt).count() < showSeconds;
        const bool reportFresh = reportAt.time_since_epoch().count() != 0 &&
                                 std::chrono::duration<float>(now - reportAt).count() < 8.f;
        if (!busy && !aiFresh && !reportFresh) {
            clear();
            return;
        }

        const float sc = scale;
        const float size = 17.f * sc;
        const float row = size * 1.35f;
        const float pad = 10.f * sc;
        const float width = panelWidth * sc;

        std::string title = "J.A.R.V.I.S. AI";
        std::uint32_t titleColor = kWhite;
        std::vector<std::string> lines;
        std::uint32_t bodyColor = kCyan;
        if (busy) {
            const int dots = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() / 400) % 4;
            title += "  ANALYSING" + std::string(static_cast<std::size_t>(dots), '.');
            titleColor = kAmber;
            lines = answer.empty() ? std::vector<std::string>{} : wrapText(answer, size, width - pad * 2.f, 12);
            bodyColor = kCyanDim;
        } else if (reportFresh && (!aiFresh || reportAt > answerAt)) {
            title = "J.A.R.V.I.S. REPORT";
            titleColor = reportHostile ? kRed : kGreen;
            lines = wrapText(reportText, size, width - pad * 2.f, 8);
            bodyColor = reportHostile ? kRed : kCyan;
        } else if (!error.empty()) {
            titleColor = kRed;
            lines = wrapText(error, size, width - pad * 2.f, 8);
            bodyColor = kRed;
        } else {
            if (answerAuto) {
                title = "J.A.R.V.I.S. INSIGHT";
                titleColor = kAmber;
            }
            lines = wrapText(answer, size, width - pad * 2.f, 12);
        }

        const std::string voiceProblem = voice ? voiceError() : std::string();
        const std::size_t problemLines = voiceProblem.empty() ? 0 : 1;
        const float height = pad * 2.f + row * (static_cast<float>(lines.size() + problemLines) + 1.f);
        Canvas canvas;
        canvas.panel(hudX, hudY, width, height, sc, opacity);
        float y = hudY + pad;
        canvas.text(hudX + pad, y, size, argb(1.f, titleColor), title);
        y += row;
        for (const auto& line : lines) {
            canvas.text(hudX + pad, y, size, argb(1.f, bodyColor), line);
            y += row;
        }
        if (!voiceProblem.empty()) {
            canvas.text(hudX + pad, y, size * 0.85f, argb(1.f, kAmber), wrapText(voiceProblem, size * 0.85f, width - pad * 2.f, 1).front());
        }
        canvas.submit(id);
    }

private:
    using Clock = std::chrono::steady_clock;

    static std::string lowerName(const std::string& type) {
        std::string out = type;
        for (auto& ch : out) ch = ch == '_' ? ' ' : static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        return out;
    }

    static std::string capitalised(std::string text) {
        if (!text.empty()) text[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(text[0])));
        return text;
    }

    bool indonesian() const { return voiceLang.rfind("id", 0) == 0; }

    static std::string direction(const char* english, bool indo) {
        if (!indo) return english;
        static const std::pair<const char*, const char*> map[] = {
            {"ahead-right", "depan kanan"}, {"ahead-left", "depan kiri"}, {"behind-right", "belakang kanan"},
            {"behind-left", "belakang kiri"}, {"ahead", "depan"}, {"behind", "belakang"}, {"right", "kanan"}, {"left", "kiri"}};
        for (const auto& entry : map) {
            if (std::string_view(english) == entry.first) return entry.second;
        }
        return english;
    }

    void announce(const std::string& text, bool hostile) {
        reportText = text;
        reportHostile = hostile;
        reportAt = Clock::now();
        if (voice) voiceSpeak(text, voiceLang);
    }

    struct Tip { std::string_view type; const char* en; const char* id; };

    // Short local advice, used when no AI insight will follow (no key, or insights switched off).
    static const char* tipFor(const std::string& type, bool indo) {
        static const Tip tips[] = {
            {"zombie", "Slow melee. Back away while hitting it.", "Serangan dekat, lambat. Mundur sambil memukul."},
            {"husk", "Causes hunger. Eat after the fight.", "Bikin lapar. Makan setelah bertarung."},
            {"drowned", "May carry a trident. Fight it on land.", "Bisa bawa trident. Lawan di darat."},
            {"skeleton", "Shoots arrows. Close in fast or hide.", "Menembak panah. Dekati cepat atau berlindung."},
            {"stray", "Slows you with arrows. Close in fast.", "Panahnya memperlambat. Dekati cepat."},
            {"creeper", "Explodes up close. Keep your distance.", "Meledak jika dekat. Jaga jarak."},
            {"spider", "Fast and climbs walls. Fight in the open.", "Cepat dan bisa memanjat. Lawan di tempat terbuka."},
            {"cave_spider", "Poisonous bite. Keep moving.", "Gigitan beracun. Terus bergerak."},
            {"enderman", "Never look at its head. Avoid it.", "Jangan menatap kepalanya. Hindari."},
            {"witch", "Throws potions. Close in or use a bow.", "Melempar ramuan. Dekati atau pakai busur."},
            {"slime", "Splits when hit. Kill the small ones last.", "Terpecah saat dipukul. Habisi yang kecil terakhir."},
            {"magma_cube", "Splits and burns. Watch your footing.", "Terpecah dan panas. Hati-hati melangkah."},
            {"phantom", "Appears when you skip sleep. Use a bed.", "Muncul jika kurang tidur. Tidurlah di kasur."},
            {"pillager", "Crossbow raider. Take cover.", "Pemanah crossbow. Berlindung."},
            {"vindicator", "Axe raider, hits hard. Keep distance.", "Pembawa kapak, sakit. Jaga jarak."},
            {"blaze", "Fireballs. Use water or snowballs.", "Menembak api. Pakai air atau bola salju."},
            {"ghast", "Fireballs can be deflected by hitting them.", "Bola api bisa dipantulkan dengan memukulnya."},
            {"warden", "Do not fight it. Sneak away.", "Jangan dilawan. Menyelinap pergi."},
            {"guardian", "Laser from the water. Break line of sight.", "Laser dari air. Hindari garis pandang."},
            {"silverfish", "Hides in stone. Kill it quickly.", "Bersembunyi di batu. Bunuh cepat."},
            {"vex", "Flies through walls. Kill it fast.", "Terbang menembus dinding. Bunuh cepat."},
        };
        for (const auto& tip : tips) {
            if (tip.type == type) return indo ? tip.id : tip.en;
        }
        return nullptr;
    }

    bool insightEnabled() const { return autoInsight && !apiKey.empty(); }

    // Remember what happened; flushInsight() sends it to the AI when the cooldown allows.
    void queueInsight(const std::string& hint, bool urgent, Clock::time_point now) {
        if (!insightEnabled()) return;
        if (!pendingHint.empty() && pendingUrgent && !urgent && now - insightAt < std::chrono::seconds(15)) return;
        pendingHint = hint;
        pendingUrgent = urgent;
        insightAt = now;
    }

    void flushInsight(Clock::time_point now) {
        if (pendingHint.empty()) return;
        if (!insightEnabled() || now - insightAt > std::chrono::seconds(15)) {
            pendingHint.clear();   // stale: the situation has probably changed
            return;
        }
        if (insightCount >= kMaxInsights) return;
        const float gap = pendingUrgent ? std::min(insightCooldown, 10.f) : insightCooldown;
        if (lastInsightAt.time_since_epoch().count() != 0 &&
            std::chrono::duration<float>(now - lastInsightAt).count() < gap) {
            return;
        }
        if (ai().busy.load()) return;

        AiJob job;
        job.provider = provider;
        job.apiKey = apiKey;
        job.model = model;
        job.language = language;
        job.question = pendingHint;
        job.speak = voice;
        job.voiceLanguage = voiceLang;
        job.proactive = true;
        std::string reason;
        if (aiRequest(job, reason)) {
            lastInsightAt = now;
            ++insightCount;
            pendingHint.clear();
        }
    }

    static bool notableBlock(const std::string& name) {
        std::string lower;
        for (char ch : name) lower.push_back(ch == '_' ? ' ' : static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        static const char* words[] = {" ore", "ancient debris", "spawner", "chest", "lava", "tnt", "magma", "cobweb",
                                      "sculk", "vault", "trial", "obsidian", "portal", "beacon", "enchanting", "anvil"};
        if (lower.size() >= 3 && lower.compare(0, 3, "ore") == 0) return true;
        for (const char* word : words) {
            if (lower.find(word) != std::string::npos) return true;
        }
        return false;
    }

    void autoReports(const Snapshot& s, Clock::time_point now) {
        if (!s.valid) return;
        const bool indo = indonesian();
        const bool tipsLocally = !insightEnabled();   // the AI adds its own tip when insights are on

        // 1) The mob under the crosshair.
        if (s.target.kind == Target::Kind::Entity && s.target.hasBox) {
            lastEntityTarget = now;
            const std::string key = s.target.type.empty() ? (s.target.player ? "player" : "mob") : s.target.type;
            if (key != targetKey) {
                targetKey = key;
                targetSince = now;
                targetReported = false;
            }
            if (!targetReported && now - targetSince > std::chrono::milliseconds(450) &&
                now - lastTargetReport > std::chrono::milliseconds(1500)) {
                targetReported = true;
                lastTargetReport = now;
                std::string name = s.target.type.empty()
                    ? (s.target.player ? std::string("Player") : (indo ? std::string("Makhluk") : std::string("Mob")))
                    : capitalised(lowerName(s.target.type));
                if (autoReport) {
                    char text[160];
                    if (s.target.hostile) {
                        std::snprintf(text, sizeof(text), indo ? "%s, musuh, %.0f meter." : "%s, hostile, %.0f meters.",
                                      name.c_str(), s.target.dist);
                    } else {
                        std::snprintf(text, sizeof(text), indo ? "%s, %.0f meter." : "%s, %.0f meters.", name.c_str(), s.target.dist);
                    }
                    std::string full = text;
                    if (tipsLocally && s.target.hostile) {
                        if (const char* tip = tipFor(s.target.type, indo)) full += std::string(" ") + tip;
                    }
                    announce(full, s.target.hostile);
                }
                // Only dangerous or unusual targets are worth an AI call; a cow is not.
                if (s.target.hostile || s.target.player) {
                    const auto seen = seenAt.find(key);
                    if (seen == seenAt.end() || now - seen->second > std::chrono::minutes(4)) {
                        seenAt[key] = now;
                        char hint[200];
                        std::snprintf(hint, sizeof(hint), "The player is aiming at %s%s, %.0f meters away.",
                                      s.target.hostile ? "a hostile " : "a ", name.c_str(), s.target.dist);
                        queueInsight(hint, s.target.hostile, now);
                    }
                }
            }
        } else if (now - lastEntityTarget > std::chrono::milliseconds(1200)) {
            targetKey.clear();
        }

        // 2) The block under the crosshair, when it is something worth knowing about.
        if (s.target.kind == Target::Kind::Block && !s.target.name.empty() && notableBlock(s.target.name)) {
            if (s.target.name != blockKey) {
                blockKey = s.target.name;
                blockSince = now;
                blockQueued = false;
            }
            if (!blockQueued && now - blockSince > std::chrono::milliseconds(1500)) {
                blockQueued = true;
                const auto seen = seenAt.find("block:" + blockKey);
                if (seen == seenAt.end() || now - seen->second > std::chrono::minutes(4)) {
                    seenAt["block:" + blockKey] = now;
                    char hint[200];
                    std::snprintf(hint, sizeof(hint), "The player is looking at the block %s, %.0f meters away.",
                                  s.target.name.c_str(), s.target.dist);
                    queueInsight(hint, false, now);
                }
            }
        } else if (s.target.kind != Target::Kind::Block) {
            blockKey.clear();
        }

        // 3) New hostiles close by.
        int count = 0;
        const Contact* nearest = nullptr;
        std::vector<std::pair<std::string, int>> groups;
        for (const auto& contact : s.contacts) {
            if (!contact.hostile || contact.dist > warnRadius) continue;
            ++count;
            if (!nearest || contact.dist < nearest->dist) nearest = &contact;
            const std::string name = contact.type.empty() ? std::string("hostile") : lowerName(contact.type);
            bool found = false;
            for (auto& group : groups) {
                if (group.first == name) { ++group.second; found = true; break; }
            }
            if (!found) groups.emplace_back(name, 1);
        }
        if (count > lastHostileCount && now - lastThreatReport > std::chrono::seconds(6) && nearest) {
            lastThreatReport = now;
            std::string list;
            for (std::size_t i = 0; i < groups.size() && i < 4; ++i) {
                if (i) list += ", ";
                list += std::to_string(groups[i].second) + " " + groups[i].first;
                if (!indo && groups[i].second > 1 && groups[i].first.back() != 's') list += "s";
            }
            const std::string where = direction(relativeName(nearest->dx, nearest->dz, s.yaw), indo);
            if (autoReport) {
                char text[220];
                std::snprintf(text, sizeof(text), indo ? "Peringatan. %s. Terdekat %.0f meter, %s."
                                                       : "Warning. %s. Nearest %.0f meters, %s.",
                              list.c_str(), nearest->dist, where.c_str());
                std::string full = text;
                if (tipsLocally) {
                    if (const char* tip = tipFor(nearest->type, indo)) full += std::string(" ") + tip;
                }
                announce(full, true);
            }
            char hint[260];
            std::snprintf(hint, sizeof(hint), "Hostile mobs just came within %.0f meters: %s. The nearest is %.0f meters away, %s.",
                          warnRadius, list.c_str(), nearest->dist, relativeName(nearest->dx, nearest->dz, s.yaw));
            queueInsight(hint, true, now);
        }
        lastHostileCount = count;

        flushInsight(now);
    }

    std::string reportText;
    bool reportHostile = false;
    Clock::time_point reportAt{};
    std::string targetKey;
    bool targetReported = false;
    Clock::time_point targetSince{};
    Clock::time_point lastEntityTarget{};
    Clock::time_point lastTargetReport{};
    Clock::time_point lastThreatReport{};
    int lastHostileCount = 0;

    static constexpr int kMaxInsights = 200;   // per session; keeps free-tier quotas safe
    std::string pendingHint;
    bool pendingUrgent = false;
    Clock::time_point insightAt{};
    Clock::time_point lastInsightAt{};
    int insightCount = 0;
    std::map<std::string, Clock::time_point> seenAt;
    std::string blockKey;
    Clock::time_point blockSince{};
    bool blockQueued = false;

    std::string pending;
    std::chrono::steady_clock::time_point pendingAt{};
    std::chrono::steady_clock::time_point createdAt = std::chrono::steady_clock::now();
};

// ===========================================================================
// Config file + Mod Menu registration
// ===========================================================================

inline Panel* findPanel(std::string_view id) {
    for (auto& panel : core().panels) {
        if (panel->id == id) return panel.get();
    }
    return nullptr;
}

inline void saveConfig() {
    auto& c = core();
    if (c.configPath.empty()) return;
    try {
        nlohmann::json root = nlohmann::json::object();
        for (const auto& panel : c.panels) {
            nlohmann::json entry = nlohmann::json::object();
            panel->toJson(entry);
            root[panel->id] = entry;
        }
        std::error_code ec;
        std::filesystem::create_directories(c.configPath.parent_path(), ec);
        std::ofstream file(c.configPath, std::ios::trunc);
        if (file) file << root.dump(2);
    } catch (...) {
    }
}

inline void loadConfig() {
    auto& c = core();
    if (c.configPath.empty()) return;
    try {
        std::ifstream file(c.configPath);
        if (!file) return;
        const nlohmann::json root = nlohmann::json::parse(file, nullptr, false);
        if (!root.is_object()) return;
        for (auto& panel : c.panels) {
            if (root.contains(panel->id)) panel->fromJson(root[panel->id]);
        }
    } catch (...) {
    }
}

inline void onModuleToggle(std::string_view moduleId, bool enabled) {
    Panel* panel = findPanel(moduleId);
    if (!panel) return;
    panel->enabled = enabled;
    panel->onEnabledChanged();
    if (!enabled) panel->clear();
    saveConfig();
}

inline void onModuleConfigChanged(std::string_view moduleId, std::string_view key, std::string_view value) {
    Panel* panel = findPanel(moduleId);
    if (!panel) return;
    if (panel->set(std::string(key), value)) saveConfig();
}

// Note: modId is deliberately NOT set. The preloader fills it in from the mod that is
// being enabled and rejects the module when a different id is declared.
inline void registerModules() {
    auto& c = core();
    if (c.registered) return;
    for (auto& panel : c.panels) {
        pl::modmenu::ModuleBuilder builder(panel->id, panel->name.c_str());
        builder.description(panel->description.c_str())
            .defaultEnabled(panel->enabled)
            .onToggle(onModuleToggle)
            .onConfigChanged(onModuleConfigChanged);
        for (const auto& entry : panel->entries()) {
            builder.config(entry.key, entry.displayName, entry.type, entry.defaultValue, entry.minValue, entry.maxValue, std::string());
        }
        if (!builder.registerModule()) {
            std::fprintf(stderr, "[Jarvis] Mod Menu rejected module %s\n", panel->id.c_str());
        }
        if (panel->id == "jarvis.ai") {
            // On-screen button: one tap asks Jarvis about whatever you are looking at (no typing).
            const bool buttonRegistered = pl::modmenu::ButtonBuilder("jarvis.ai.scan", "Ask Jarvis")
                .moduleId(panel->id)
                .label("Ask Jarvis")
                .behavior(pl::modmenu::ButtonBehavior::Click)
                .stylePreset(pl::modmenu::ButtonStylePreset::Keycap)   // the blocky Minecraft key look
                .styleColors(0xFF2E6FE0, 0xFF5CA0FF, 0xFF16398A)       // blue: normal, pressed, border
                .textColor(0xFFFFFFFF)
                .activeTextColor(0xFFFFFFFF)
                .sizeScale(2.2f, 1.0f)
                .onEvent([](std::string_view, pl::modmenu::ButtonEvent event, float) {
                    if (event != pl::modmenu::ButtonEvent::Click) return;
                    if (!core().started) return;
                    if (auto* aiPanel = dynamic_cast<AiPanel*>(findPanel("jarvis.ai"))) {
                        if (aiPanel->enabled) aiPanel->ask();
                    }
                })
                .registerButton();
            if (!buttonRegistered) std::fprintf(stderr, "[Jarvis] Mod Menu rejected the AI button\n");
        }
    }
    c.registered = true;
}

// ===========================================================================
// Game hook (ClientInstance::update runs once per frame on the game thread)
// ===========================================================================

inline void drawAll() {
    auto& c = core();
    bool any = false;
    for (const auto& panel : c.panels) any = any || panel->enabled;
    if (!any) return;
    const Snapshot s = snapshot();
    for (auto& panel : c.panels) {
        if (panel->enabled) panel->draw(s);
    }
}

inline void* clientUpdateDetour(void* clientInstance, bool value) {
    auto& c = core();
    void* result = c.updateOriginal ? c.updateOriginal(clientInstance, value) : nullptr;
    if (!c.active.load(std::memory_order_acquire) || !clientInstance) return result;

    void* player = nullptr;
    if (c.getLocalPlayer) {
        player = c.getLocalPlayer(clientInstance);
    } else {
        void** vtable = vtableOf(clientInstance);
        const auto slot = jsdk::offsets::VTable::ClientInstanceGetLocalPlayer;
        if (vtable && vtable[slot]) player = reinterpret_cast<void* (*)(void*)>(vtable[slot])(clientInstance);
    }
    if (player) onTick(reinterpret_cast<jsdk::Player*>(player));
    drawAll();
    return result;
}

inline bool libraryLoaded(const char* name) {
    void* handle = dlopen(name, RTLD_NOW | RTLD_NOLOAD);
    if (!handle) return false;
    dlclose(handle);
    return true;
}

// The game library may not be loaded yet when the mod is enabled, so wait for it
// (up to about five minutes), scan once, then install the hook.
inline void attachLoop() {
    auto& c = core();
    constexpr const char* kLibrary = "libminecraftpe.so";
    for (int attempt = 0; attempt < 600 && !c.stopping.load(); ++attempt) {
        if (libraryLoaded(kLibrary)) {
            if (!c.resolved) {
                c.resolved = jsdk::resolveAll(kLibrary);
                if (!c.resolved) {
                    c.state = 2;   // patterns do not match this game version
                    return;
                }
            }
            bindSignatures();
            c.hookTarget = reinterpret_cast<void*>(jsdk::address(jsdk::Sig::ClientInstanceUpdate));
            if (pl::memory::hook(c.hookTarget, reinterpret_cast<void*>(clientUpdateDetour),
                                 reinterpret_cast<void**>(&c.updateOriginal)) != 0) {
                c.hookTarget = nullptr;
                c.state = 3;
                return;
            }
            c.hooked = true;
            c.state = 1;
            c.active.store(true, std::memory_order_release);
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

}  // namespace detail

// ===========================================================================
// Public API
// ===========================================================================

// modId / configDir: pass getSelf().getId() and getSelf().getConfigDir().
template <class Id, class Dir>
inline void start(const Id& modId, const Dir& configDir) {
    auto& c = detail::core();
    if (c.started) return;
    c.started = true;
    c.stopping = false;
    c.modId = std::string(modId);
    c.configPath = std::filesystem::path(configDir) / "jarvis.json";

    if (c.panels.empty()) {
        c.panels.push_back(std::make_shared<detail::StatusPanel>());
        c.panels.push_back(std::make_shared<detail::RadarPanel>());
        c.panels.push_back(std::make_shared<detail::ScannerPanel>());
        c.panels.push_back(std::make_shared<detail::GlowPanel>());
        c.panels.push_back(std::make_shared<detail::AiPanel>());
    }
    detail::loadConfig();
    for (auto& panel : c.panels) panel->onEnabledChanged();
    detail::registerModules();

    if (c.attachThread.joinable()) c.attachThread.join();
    c.attachThread = std::thread(detail::attachLoop);
}

inline void stop() {
    auto& c = detail::core();
    if (!c.started) return;
    c.stopping = true;
    if (c.attachThread.joinable()) c.attachThread.join();
    c.active.store(false, std::memory_order_release);
    if (c.hooked && c.hookTarget) {
        pl::memory::unhook(c.hookTarget, reinterpret_cast<void*>(detail::clientUpdateDetour));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));   // let an in-flight call finish
    }
    c.hooked = false;
    c.hookTarget = nullptr;
    c.state = 0;
    detail::aiShutdown();
    detail::voiceShutdown();
    detail::core().settings.aiNeeds = false;
    detail::core().settings.needGlow = false;
    for (auto& panel : c.panels) panel->clear();
    detail::saveConfig();
    c.started = false;
}

}  // namespace jarvis
